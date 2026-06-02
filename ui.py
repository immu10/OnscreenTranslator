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

from PyQt6.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QIcon, QPixmap, QShortcut, QKeySequence


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
FONT_FAMILY = "Malgun Gothic"  # ships with Windows, renders Korean well


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

        outline_pen = QPen(QColor(0, 255, 80, 220), 2)
        bg_color = QColor(0, 0, 0, 200)
        text_color = QColor(255, 255, 255, 255)

        win_w = self.width()
        win_h = self.height()

        for box, text, trans in self._snapshot:
            (x1, y1), (x2, y2) = box
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            if box_w == 0 or box_h == 0:
                continue

            # Draw box outline (helps see what was detected)
            painter.setPen(outline_pen)
            painter.drawRect(x1, y1, box_w, box_h)

            label = trans or text
            if not label:
                continue

            # Font size scales with box height; clamped to a usable range.
            font_pt = max(9, min(20, int(box_h * 0.35)))
            font = QFont(FONT_FAMILY, font_pt)
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


def run(monitor_index, region, results_getter, stop_event):
    """Blocks until stop_event is set or the Qt app quits.

    monitor_index: which screen the overlay covers (matches dxcam.output_idx).
    region:        (x1, y1, x2, y2) in monitor-local coordinates, matching what
                   the capture stream is cropped to. Pass None for full monitor.
    results_getter: callable returning the current list of (box, ko, en).
    stop_event:    threading.Event — when set, the overlay closes and run()
                   returns.
    """
    app = QApplication.instance() or QApplication([])

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

    def quit_app():
        stop_event.set()
        app.quit()

    # System tray icon: right-click for menu, always-on affordance to close.
    tray = QSystemTrayIcon()
    pix = QPixmap(16, 16)
    pix.fill(QColor(0, 200, 80))
    tray.setIcon(QIcon(pix))
    tray.setToolTip("Korean OCR Translator — right-click to quit")
    tray_menu = QMenu()
    tray_menu.addAction("Quit").triggered.connect(quit_app)
    tray.setContextMenu(tray_menu)
    tray.show()
    # Keep a reference to tray so it doesn't get GC'd (would hide the icon).
    overlay._tray = tray

    # Esc as a backup quit hotkey (works while overlay is the focused window —
    # since it's click-through, focus rarely lands on it, so tray is primary).
    QShortcut(QKeySequence("Esc"), overlay).activated.connect(quit_app)

    # Poll stop_event so external threads can ask us to quit.
    poll = QTimer()
    poll.timeout.connect(lambda: app.quit() if stop_event.is_set() else None)
    poll.start(200)

    app.exec()
