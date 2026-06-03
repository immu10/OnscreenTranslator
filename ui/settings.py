"""User-tweakable settings, persisted to settings.json.

Module-level `SETTINGS` dict is the live source of truth — both ui.py and
ocr_worker read from it on every iteration, so changes made through the
SettingsDialog take effect immediately (except if and where labeled "(restart)").
"""

import ctypes
import json
import os

from paths import settings_file

from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QColorDialog, QCheckBox, QDialogButtonBox, QApplication,
    QLabel, QWidget, QHBoxLayout,
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QPainter, QPen, QFont

DEFAULTS = {
    "monitor_index": 1,           # which screen to capture
    "crop_top_ratio": 0.10,       # fallback if custom_region is null
    "crop_bottom_ratio": 0.10,    # fallback if custom_region is null
    "custom_region": None,        # [x1,y1,x2,y2] in monitor-local pixels;
                                  # overrides crop_*_ratio when set

    "box_color": [0, 200, 80, 220],     # RGBA outline around detected boxes
    "bg_color": [0, 0, 0, 200],         # behind translation text
    "text_color": [255, 255, 255, 255], # translation text
    "show_box_outline": True,

    "font_family": "Malgun Gothic",
    "font_size_min": 9,
    "font_size_max": 20,

    "ocr_upscale": 2.0,           # crop scale factor before recognition (live)
}

SETTINGS = dict(DEFAULTS)


def load():
    """Read settings.json into SETTINGS (only known keys, default-fallback)."""
    path = settings_file()
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        for k in DEFAULTS:
            if k in loaded:
                SETTINGS[k] = loaded[k]
        print(f"[settings] loaded from {path}", flush=True)
    except Exception as e:
        print(f"[settings] load failed: {e}", flush=True)


def save():
    """Write current SETTINGS to settings.json."""
    path = settings_file()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SETTINGS, f, indent=2, ensure_ascii=False)
        print(f"[settings] saved to {path}", flush=True)
    except Exception as e:
        print(f"[settings] save failed: {e}", flush=True)


_WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _exclude_widget_from_capture(widget):
    """Hide widget from dxcam / Print Screen / OBS etc. Win10 build 19041+."""
    try:
        hwnd = int(widget.winId())
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAPTURE)
    except Exception:
        pass


class RegionPicker(QWidget):
    """Fullscreen translucent overlay that lets the user drag a rectangle to
    pick a capture region. Calls on_done(region_or_None) when finished.
    Region is (x1, y1, x2, y2) in monitor-local pixels."""

    def __init__(self, monitor_index, on_done):
        super().__init__()
        self._on_done = on_done
        self._start = None
        self._end = None

        screens = QApplication.instance().screens()
        scr = (screens[monitor_index]
               if 0 <= monitor_index < len(screens) else screens[0])
        sg = scr.geometry()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(sg.x(), sg.y(), sg.width(), sg.height())
        self.setCursor(Qt.CursorShape.CrossCursor)

    def showEvent(self, event):
        _exclude_widget_from_capture(self)
        self.activateWindow()
        self.raise_()
        super().showEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dim entire monitor
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self._start is not None and self._end is not None:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            x2 = max(self._start.x(), self._end.x())
            y2 = max(self._start.y(), self._end.y())
            sel = QRect(x1, y1, x2 - x1, y2 - y1)

            # Cut the selection out of the dim layer
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(sel, Qt.GlobalColor.transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Outline
            p.setPen(QPen(QColor(0, 200, 80, 230), 2))
            p.drawRect(sel)

            # Size label
            p.setPen(QColor(255, 255, 255))
            label_font = QFont("Segoe UI", 11)
            label_font.setBold(True)
            p.setFont(label_font)
            p.drawText(x1 + 8, max(y1 - 8, 20), f"{x2 - x1} x {y2 - y1}")

        # Instructions
        p.setPen(QColor(255, 255, 255, 220))
        instr_font = QFont("Segoe UI", 14)
        instr_font.setBold(True)
        p.setFont(instr_font)
        p.drawText(
            self.rect().adjusted(0, 24, 0, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "Drag to select capture region.   ESC to cancel.",
        )

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._start = e.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self._end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._start and self._end:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            x2 = max(self._start.x(), self._end.x())
            y2 = max(self._start.y(), self._end.y())
            if x2 - x1 > 10 and y2 - y1 > 10:
                self._on_done([x1, y1, x2, y2])
            else:
                self._on_done(None)
            self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._on_done(None)
            self.close()


class _ColorButton(QPushButton):
    """Button showing a color swatch; click opens a color picker with alpha."""

    def __init__(self, key):
        super().__init__()
        self.key = key
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        r, g, b, a = SETTINGS[self.key]
        self.setText(f"rgba({r},{g},{b},{a})")
        # Approximate the alpha visually with a lighter shade.
        css_alpha = a / 255
        fg = "white" if (r + g + b) / 3 < 128 else "black"
        self.setStyleSheet(
            f"background-color: rgba({r},{g},{b},{css_alpha:.2f});"
            f" color: {fg}; padding: 6px;"
        )

    def _pick(self):
        cur = QColor(*SETTINGS[self.key])
        new = QColorDialog.getColor(
            cur, self, "Pick color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if new.isValid():
            SETTINGS[self.key] = [new.red(), new.green(), new.blue(), new.alpha()]
            self._refresh()


class SettingsDialog(QDialog):
    def __init__(self, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle("Translator Settings")
        self._on_apply = on_apply
        layout = QFormLayout(self)

        # Monitor (restart-required)
        self.monitor_cb = QComboBox()
        for i, screen in enumerate(QApplication.instance().screens()):
            g = screen.geometry()
            self.monitor_cb.addItem(
                f"#{i}  {g.width()}x{g.height()}  @ ({g.x()}, {g.y()})", i,
            )
        for i in range(self.monitor_cb.count()):
            if self.monitor_cb.itemData(i) == SETTINGS["monitor_index"]:
                self.monitor_cb.setCurrentIndex(i)
                break
        layout.addRow("Monitor:", self.monitor_cb)

        # Crop ratios (restart-required)
        self.crop_top = QDoubleSpinBox()
        self.crop_top.setRange(0.0, 0.4); self.crop_top.setSingleStep(0.05)
        self.crop_top.setValue(SETTINGS["crop_top_ratio"])
        layout.addRow("Crop top %:", self.crop_top)

        self.crop_bot = QDoubleSpinBox()
        self.crop_bot.setRange(0.0, 0.4); self.crop_bot.setSingleStep(0.05)
        self.crop_bot.setValue(SETTINGS["crop_bottom_ratio"])
        layout.addRow("Crop bottom %:", self.crop_bot)

        # Custom region (overrides crop %): pick by drag, or clear.
        self.region_label = QLabel(self._region_text())
        region_row = QWidget()
        region_layout = QHBoxLayout(region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        btn_pick = QPushButton("Pick from screen...")
        btn_pick.clicked.connect(self._pick_region)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_region)
        region_layout.addWidget(self.region_label, stretch=1)
        region_layout.addWidget(btn_pick)
        region_layout.addWidget(btn_clear)
        layout.addRow("Custom region:", region_row)

        # Live: colors
        self.box_color = _ColorButton("box_color")
        layout.addRow("Box outline color:", self.box_color)

        self.bg_color = _ColorButton("bg_color")
        layout.addRow("Text background:", self.bg_color)

        self.text_color = _ColorButton("text_color")
        layout.addRow("Text color:", self.text_color)

        # Live: outline toggle
        self.outline_cb = QCheckBox()
        self.outline_cb.setChecked(SETTINGS["show_box_outline"])
        self.outline_cb.toggled.connect(
            lambda v: SETTINGS.__setitem__("show_box_outline", v)
        )
        layout.addRow("Show box outline:", self.outline_cb)

        # Live: font sizes
        self.font_min = QSpinBox()
        self.font_min.setRange(6, 40)
        self.font_min.setValue(SETTINGS["font_size_min"])
        self.font_min.valueChanged.connect(
            lambda v: SETTINGS.__setitem__("font_size_min", v)
        )
        layout.addRow("Font min pt:", self.font_min)

        self.font_max = QSpinBox()
        self.font_max.setRange(6, 60)
        self.font_max.setValue(SETTINGS["font_size_max"])
        self.font_max.valueChanged.connect(
            lambda v: SETTINGS.__setitem__("font_size_max", v)
        )
        layout.addRow("Font max pt:", self.font_max)

        # Live: OCR upscale
        self.upscale = QDoubleSpinBox()
        self.upscale.setRange(1.0, 5.0)
        self.upscale.setSingleStep(0.5)
        self.upscale.setValue(SETTINGS["ocr_upscale"])
        self.upscale.valueChanged.connect(
            lambda v: SETTINGS.__setitem__("ocr_upscale", v)
        )
        layout.addRow("OCR upscale (live):", self.upscale)

        layout.addRow(QLabel("<i>All changes apply on Save.</i>"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._save)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addRow(buttons)

    def _region_text(self):
        r = SETTINGS.get("custom_region")
        if r is None:
            return "<i>(none — using crop %)</i>"
        return (f"({r[0]},{r[1]}) → ({r[2]},{r[3]})  "
                f"[{r[2]-r[0]} × {r[3]-r[1]}]")

    def _pick_region(self):
        # Save current monitor choice first so the picker shows on the right one.
        SETTINGS["monitor_index"] = self.monitor_cb.currentData()
        self.hide()  # get the dialog out of the way during pick
        self._picker = RegionPicker(
            SETTINGS["monitor_index"],
            on_done=self._on_region_picked,
        )
        self._picker.show()

    def _on_region_picked(self, region):
        self.show()
        self.raise_()
        if region is not None:
            SETTINGS["custom_region"] = region
            self.region_label.setText(self._region_text())

    def _clear_region(self):
        SETTINGS["custom_region"] = None
        self.region_label.setText(self._region_text())

    def _save(self):
        # Pull values from the restart-flagged widgets (they aren't connected
        # via live signals because they need an explicit apply).
        SETTINGS["monitor_index"] = self.monitor_cb.currentData()
        SETTINGS["crop_top_ratio"] = self.crop_top.value()
        SETTINGS["crop_bottom_ratio"] = self.crop_bot.value()
        save()
        if self._on_apply is not None:
            try:
                self._on_apply()
            except Exception as e:
                print(f"[settings] on_apply failed: {e}", flush=True)
        self.close()
