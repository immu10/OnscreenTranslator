"""Live log viewer.

`install_tap()` wraps sys.stdout/sys.stderr at import time so every `print(...)`
in the app is mirrored into an in-memory ring buffer (and still written to the
real console). `LogViewer` is a QDialog that polls the buffer and renders new
lines into a read-only text widget — opened from the floating-button / tray
"Logs..." action.

Keep this import-light: install_tap() must be safe to call before QApplication
exists. Qt widget classes are defined at module level (PyQt6.QtWidgets is
already imported by ui.settings), but no QObject is *constructed* until the
user actually opens the viewer.
"""

import collections
import sys
import threading

from PyQt6.QtWidgets import (
    QDialog, QPlainTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QLabel,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

MAX_LINES = 2000

_buffer = collections.deque(maxlen=MAX_LINES)
_lock = threading.Lock()
_seq = 0           # monotonically increases as lines are appended
_installed = False


class _Tee:
    """File-like wrapper that forwards every write to the original stream and
    also splits the text into lines stored in the shared ring buffer."""

    def __init__(self, original, tag):
        self._orig = original
        self._tag = tag
        self._pending = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        if not isinstance(s, str):
            return
        global _seq
        with _lock:
            self._pending += s
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                _buffer.append(line)
                _seq += 1

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._orig.fileno()


def install_tap():
    """Idempotently wrap sys.stdout / sys.stderr."""
    global _installed
    if _installed:
        return
    sys.stdout = _Tee(sys.stdout, "out")
    sys.stderr = _Tee(sys.stderr, "err")
    _installed = True


def snapshot():
    """Return (seq, lines_copy). seq is the index of the LAST line in the
    returned list, so a caller can pass it back as `since` next poll."""
    with _lock:
        return _seq, list(_buffer)


def since(prev_seq):
    """Return (new_seq, [lines_added_after_prev_seq])."""
    with _lock:
        if prev_seq >= _seq:
            return _seq, []
        # How many new lines? Cap at buffer length (we may have lost some to
        # the ring's drop-oldest if the viewer was slow).
        new_count = min(_seq - prev_seq, len(_buffer))
        if new_count == 0:
            return _seq, []
        lines = list(_buffer)[-new_count:]
        return _seq, lines


class LogViewer(QDialog):
    """Read-only live tail of stdout/stderr. Polls the ring buffer at 5 Hz."""

    POLL_MS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Translator Logs")
        self.resize(820, 480)

        layout = QVBoxLayout(self)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.view.setFont(mono)
        self.view.setMaximumBlockCount(MAX_LINES)
        layout.addWidget(self.view, stretch=1)

        # Bottom bar: autoscroll toggle + Clear + Copy + Close
        bar = QHBoxLayout()
        self.autoscroll_cb = QCheckBox("Auto-scroll")
        self.autoscroll_cb.setChecked(True)
        bar.addWidget(self.autoscroll_cb)
        bar.addStretch(1)
        self.status = QLabel("")
        bar.addWidget(self.status)
        btn_clear = QPushButton("Clear view")
        btn_clear.clicked.connect(self._clear_view)
        bar.addWidget(btn_clear)
        btn_copy = QPushButton("Copy all")
        btn_copy.clicked.connect(self._copy_all)
        bar.addWidget(btn_copy)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        layout.addLayout(bar)

        # Seed with whatever is already in the buffer, then poll for new lines.
        seq, lines = snapshot()
        self._seq = seq
        if lines:
            self.view.setPlainText("\n".join(lines))
            self._scroll_to_end()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.POLL_MS)

    def _tick(self):
        self._seq, new_lines = since(self._seq)
        if new_lines:
            for line in new_lines:
                self.view.appendPlainText(line)
            if self.autoscroll_cb.isChecked():
                self._scroll_to_end()
        self.status.setText(f"{self._seq} lines total")

    def _scroll_to_end(self):
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_view(self):
        self.view.clear()

    def _copy_all(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.view.toPlainText())


_viewer_ref = {}  # module-level singleton holder


def open_logs(parent=None):
    """Show the log viewer; reuses the existing instance if one is open."""
    existing = _viewer_ref.get("v")
    if existing is not None and existing.isVisible():
        existing.raise_()
        existing.activateWindow()
        return existing
    v = LogViewer(parent)
    # Exclude from screen capture so its contents don't get re-OCR'd.
    try:
        from .settings import _exclude_widget_from_capture
        v.show()
        _exclude_widget_from_capture(v)
    except Exception:
        v.show()
    _viewer_ref["v"] = v
    return v
