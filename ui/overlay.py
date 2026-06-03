"""Click-through transparent overlay that draws translations beside detected
text boxes. Floats over whatever's on screen; mouse events pass through to
the application underneath.

Usage from main.py:
    import ui
    ui.run(monitor_index=1, region=(0, 108, 1920, 972),
           results_getter=get_results, stop_event=stop_event)

`results_getter` must return a list of ((x1,y1),(x2,y2), korean, english) entries
in *capture-local* coordinates (matching the region the detector sees).
"""

import ctypes

from PyQt6.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu,
    QPushButton, QVBoxLayout, QLabel,
)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QIcon, QPixmap, QShortcut, QKeySequence

from .settings import SettingsDialog, SETTINGS
from .logs import open_logs


# Win32 SetWindowDisplayAffinity flag: window is visible to user but invisible
# to ALL screen-capture APIs (dxcam, OBS, Print Screen, etc).
# Windows 10 version 2004+ only.
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _exclude_from_capture(widget):
    """Tell Windows that this window must not appear in any screen capture.
    Prevents the dxcam pipeline from re-OCRing our own drawn translations."""
    try:
        hwnd = int(widget.winId())
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(
            hwnd, WDA_EXCLUDEFROMCAPTURE
        )
        if ok:
            print(f"[ui] overlay excluded from screen capture (hwnd={hwnd})",
                  flush=True)
        else:
            err = ctypes.get_last_error()
            print(f"[ui] SetWindowDisplayAffinity failed (err={err}) — "
                  "needs Windows 10 build 19041+", flush=True)
    except Exception as e:
        print(f"[ui] could not exclude window from capture: {e}", flush=True)

REPAINT_HZ = 30
LABEL_PAD = 6           # gap between source box and translation label


class Overlay(QWidget):
    def __init__(self, x, y, w, h, results_getter):
        super().__init__()
        self.results_getter = results_getter
        self._snapshot = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                       # no taskbar, no alt-tab
            | Qt.WindowType.WindowTransparentForInput  # click-through
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(x, y, w, h)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / REPAINT_HZ))

    def _tick(self):
        try:
            self._snapshot = list(self.results_getter())
        except Exception:
            self._snapshot = []
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Live settings — read each paint so changes take effect immediately.
        outline_pen = QPen(QColor(*SETTINGS["box_color"]), 2)
        bg_color = QColor(*SETTINGS["bg_color"])
        text_color = QColor(*SETTINGS["text_color"])
        show_outline = SETTINGS["show_box_outline"]
        font_family = SETTINGS["font_family"]
        font_min = SETTINGS["font_size_min"]
        font_max = SETTINGS["font_size_max"]

        win_w = self.width()
        win_h = self.height()

        for box, text, trans in self._snapshot:
            (x1, y1), (x2, y2) = box
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            if box_w == 0 or box_h == 0:
                continue

            # Draw box outline (helps see what was detected)
            if show_outline:
                painter.setPen(outline_pen)
                painter.drawRect(x1, y1, box_w, box_h)

            label = trans or text
            if not label:
                continue

            # Font size scales with box height; clamped to a usable range.
            font_pt = max(font_min, min(font_max, int(box_h * 0.35)))
            font = QFont(font_family, font_pt)
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()

            # Wrap label width to box width if reasonable, else leave one line.
            target_w = max(160, box_w)
            text_rect = metrics.boundingRect(
                QRect(0, 0, target_w, 10_000),
                Qt.TextFlag.TextWordWrap,
                label,
            )
            text_w = text_rect.width()
            text_h = text_rect.height()

            # Prefer placement to the RIGHT of the box (centered vertically).
            tx = x2 + LABEL_PAD
            ty = y1 + (box_h - text_h) // 2

            # If right side overflows, try LEFT.
            if tx + text_w + 6 > win_w:
                tx = x1 - LABEL_PAD - text_w
            # If left also overflows, place BELOW.
            if tx < 0:
                tx = x1
                ty = y2 + LABEL_PAD
                # If below overflows, place ABOVE.
                if ty + text_h > win_h:
                    ty = y1 - LABEL_PAD - text_h

            # Final clamp so the label is always on-screen.
            tx = max(0, min(tx, win_w - text_w - 6))
            ty = max(0, min(ty, win_h - text_h - 2))

            # Background panel for readability.
            painter.fillRect(
                QRect(tx - 4, ty - 2, text_w + 8, text_h + 4),
                bg_color,
            )
            # The text itself.
            painter.setPen(text_color)
            painter.drawText(
                QRect(tx, ty, text_w, text_h),
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft,
                label,
            )


class FloatingButton(QWidget):
    """Discord-overlay-style floating handle: small always-on-top circle that
    can be dragged anywhere on screen. Click (or right-click) for a menu with
    Settings / Quit. No taskbar entry, no alt-tab — just floats."""

    SIZE = 44

    def __init__(self, on_settings, on_logs, on_quit,
                 on_pause_toggle=None, is_paused=None):
        super().__init__()
        self.on_settings = on_settings
        self.on_logs = on_logs
        self.on_quit = on_quit
        self.on_pause_toggle = on_pause_toggle
        self.is_paused = is_paused or (lambda: False)
        self._drag_offset = None
        self._dragged = False
        # Defer single-click-opens-menu so it doesn't fire when the user
        # is actually performing a double-click (which toggles pause).
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._deferred_click)
        self._pending_click_pos = None
        self._just_double_clicked = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                # no taskbar, no alt-tab
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self.SIZE, self.SIZE)

        # Default position: top-right of primary screen, just below corner.
        primary = QApplication.primaryScreen().geometry()
        self.move(primary.x() + primary.width() - self.SIZE - 24,
                  primary.y() + 24)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        paused = self.is_paused()
        ring_color = QColor(140, 140, 140, 230) if paused else QColor(0, 200, 80, 230)
        glyph_color = QColor(180, 180, 180, 240) if paused else QColor(255, 255, 255, 240)

        # Translucent dark fill, accent ring.
        p.setBrush(QColor(20, 20, 20, 210))
        p.setPen(QPen(ring_color, 2))
        p.drawEllipse(2, 2, self.SIZE - 4, self.SIZE - 4)

        # 'T' glyph (or pause bars when paused) in the center.
        p.setPen(glyph_color)
        if paused:
            # Two short vertical bars.
            cx = self.SIZE // 2
            cy = self.SIZE // 2
            bw, bh, gap = 4, 16, 4
            p.fillRect(cx - gap - bw, cy - bh // 2, bw, bh, glyph_color)
            p.fillRect(cx + gap,       cy - bh // 2, bw, bh, glyph_color)
        else:
            f = QFont("Segoe UI", 16)
            f.setBold(True)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "T")

    def _show_menu(self, global_pos):
        menu = QMenu(self)
        pause_label = "Resume" if self.is_paused() else "Pause"
        if self.on_pause_toggle is not None:
            menu.addAction(pause_label).triggered.connect(self._toggle_pause)
            menu.addSeparator()
        menu.addAction("Settings...").triggered.connect(self.on_settings)
        menu.addAction("Logs...").triggered.connect(self.on_logs)
        menu.addSeparator()
        menu.addAction("Quit").triggered.connect(self.on_quit)
        menu.exec(global_pos)

    def _toggle_pause(self):
        if self.on_pause_toggle is not None:
            self.on_pause_toggle()
            self.update()  # repaint with new state

    def _deferred_click(self):
        # Single-click survived the double-click window — open the menu.
        if self._pending_click_pos is not None:
            self._show_menu(self._pending_click_pos)
            self._pending_click_pos = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.pos()
            self._dragged = False
        elif e.button() == Qt.MouseButton.RightButton:
            # Right-click bypasses the double-click defer — open immediately.
            self._click_timer.stop()
            self._pending_click_pos = None
            self._show_menu(e.globalPosition().toPoint())

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None:
            new_pos = e.globalPosition().toPoint() - self._drag_offset
            self.move(new_pos)
            self._dragged = True
            # A drag started — cancel any pending click action.
            self._click_timer.stop()
            self._pending_click_pos = None

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._just_double_clicked:
                # This release is the tail of a double-click — don't re-arm
                # the deferred menu-open or it'll fire 250 ms later.
                self._just_double_clicked = False
            elif not self._dragged:
                # Defer the menu-open so a follow-up double-click can pre-empt it.
                self._pending_click_pos = e.globalPosition().toPoint()
                interval = QApplication.doubleClickInterval()
                self._click_timer.start(interval)
            self._drag_offset = None
            self._dragged = False

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            # Pre-empt the deferred single-click menu open AND mark so the
            # following mouseReleaseEvent doesn't re-arm it.
            self._click_timer.stop()
            self._pending_click_pos = None
            self._just_double_clicked = True
            self._toggle_pause()


def run(monitor_index, region, results_getter, stop_event, restart_capture=None,
        on_pause_toggle=None, is_paused=None):
    """Blocks until stop_event is set or the Qt app quits.

    monitor_index: which screen the overlay covers (matches dxcam.output_idx).
    region:        (x1, y1, x2, y2) in monitor-local coordinates, matching what
                   the capture stream is cropped to. Pass None for full monitor.
    results_getter: callable returning the current list of (box, ko, en).
    stop_event:    threading.Event — when set, the overlay closes and run()
                   returns.
    """
    app = QApplication.instance() or QApplication([])
    # Don't let Qt quit when the settings dialog closes — only our explicit
    # quit_app() (tray menu / handle right-click / closeEvent on a real window)
    # should end the loop.
    app.setQuitOnLastWindowClosed(False)

    screens = app.screens()
    if not screens:
        raise RuntimeError("no screens available")
    screen = screens[monitor_index] if 0 <= monitor_index < len(screens) else screens[0]
    sg = screen.geometry()

    if region is None:
        x, y, w, h = sg.x(), sg.y(), sg.width(), sg.height()
    else:
        rx1, ry1, rx2, ry2 = region
        x = sg.x() + rx1
        y = sg.y() + ry1
        w = rx2 - rx1
        h = ry2 - ry1

    print(f"[ui] overlay at screen=({x},{y}) size=({w}x{h})", flush=True)

    overlay = Overlay(x, y, w, h, results_getter)
    overlay.show()
    _exclude_from_capture(overlay)

    refs = {}  # hold strong refs so dialogs/icons aren't garbage-collected

    def quit_app():
        stop_event.set()
        app.quit()

    def reposition_overlay():
        """Move the overlay to match the current settings' monitor + region."""
        mi = SETTINGS["monitor_index"]
        screens_now = app.screens()
        if not screens_now:
            return
        scr = screens_now[mi] if 0 <= mi < len(screens_now) else screens_now[0]
        sg2 = scr.geometry()
        new_region = getattr(reposition_overlay, "_region", None)
        if new_region is None:
            return
        rx1, ry1, rx2, ry2 = new_region
        overlay.setGeometry(
            sg2.x() + rx1, sg2.y() + ry1, rx2 - rx1, ry2 - ry1,
        )
        # Re-apply capture exclusion since the HWND may have been re-realized.
        _exclude_from_capture(overlay)
        print(f"[ui] overlay moved to monitor #{mi} "
              f"at ({sg2.x()+rx1},{sg2.y()+ry1}) "
              f"size ({rx2-rx1}x{ry2-ry1})", flush=True)

    def on_settings_applied():
        """Called after the user clicks Save in SettingsDialog."""
        if restart_capture is None:
            return
        new_region = restart_capture(
            SETTINGS["monitor_index"],
            SETTINGS["crop_top_ratio"],
            SETTINGS["crop_bottom_ratio"],
            SETTINGS.get("custom_region"),
        )
        if new_region is not None:
            reposition_overlay._region = new_region
            reposition_overlay()

    def open_settings():
        dlg = SettingsDialog(on_apply=on_settings_applied)
        refs["settings_dlg"] = dlg
        dlg.show()

    def open_log_viewer():
        refs["log_dlg"] = open_logs()

    # Floating handle (Discord-overlay style) — drag to reposition, click for menu.
    handle = FloatingButton(
        on_settings=open_settings, on_logs=open_log_viewer, on_quit=quit_app,
        on_pause_toggle=on_pause_toggle, is_paused=is_paused,
    )
    _exclude_from_capture(handle)
    handle.show()
    refs["handle"] = handle

    # System tray icon — secondary affordance in case the handle is hidden behind
    # a full-screen window or accidentally dragged off-screen.
    tray = QSystemTrayIcon()
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 200, 80))
    tray.setIcon(QIcon(pix))
    tray.setToolTip("Korean OCR Translator — right-click to open menu")
    def _toggle_pause_from_tray():
        if on_pause_toggle is not None:
            on_pause_toggle()
            handle.update()  # keep handle visual in sync

    def _rebuild_tray_menu():
        tray_menu = QMenu()
        if on_pause_toggle is not None:
            label = "Resume" if (is_paused and is_paused()) else "Pause"
            tray_menu.addAction(label).triggered.connect(_toggle_pause_from_tray)
            tray_menu.addSeparator()
        tray_menu.addAction("Settings...").triggered.connect(open_settings)
        tray_menu.addAction("Logs...").triggered.connect(open_log_viewer)
        tray_menu.addSeparator()
        tray_menu.addAction("Show handle").triggered.connect(
            lambda: (handle.show(), handle.raise_())
        )
        tray_menu.addAction("Quit").triggered.connect(quit_app)
        tray.setContextMenu(tray_menu)
        refs["tray_menu"] = tray_menu  # keep alive

    _rebuild_tray_menu()
    # Rebuild on every show so the Pause/Resume label is always current.
    tray.activated.connect(lambda _reason: _rebuild_tray_menu())
    tray.show()
    refs["tray"] = tray

    # Esc as a backup quit hotkey (works while overlay is the focused window —
    # since it's click-through, focus rarely lands on it, so tray is primary).
    QShortcut(QKeySequence("Esc"), overlay).activated.connect(quit_app)

    # Poll stop_event so external threads can ask us to quit.
    poll = QTimer()
    poll.timeout.connect(lambda: app.quit() if stop_event.is_set() else None)
    poll.start(200)

    app.exec()
