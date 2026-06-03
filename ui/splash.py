"""Tiny tkinter splash window shown while the LLM loads.

Tkinter is used (not PyQt6) because Qt's graphics plugin would grab the GPU
context and crash bitsandbytes mid-load on Windows. Tkinter is pure GDI and
plays nice.
"""

import re
import threading
import tkinter as tk


# Matches tqdm's "Description: NN%|####    | cur/total [elapsed<eta, rate]"
_TQDM_RE = re.compile(
    r"^(?P<desc>.+?):\s+"
    r"(?P<pct>\d+)%\|[^|]*\|\s*"
    r"(?P<cur>\S+)/(?P<total>\S+)\s*"
    r"\[(?P<elapsed>[^<]+)<(?P<eta>[^,\]]+)"
)


def _format_line(line):
    """Return (status, progress) for the splash. If the line is a tqdm bar,
    we strip the ugly hash bar and surface just the meaningful numbers.
    Otherwise the line goes whole into `status` and `progress` is blank."""
    if not line:
        return "", ""
    m = _TQDM_RE.match(line.strip())
    if not m:
        return line, ""
    desc = m.group("desc").strip()
    progress = (
        f"{m.group('pct')}%   "
        f"{m.group('cur')}/{m.group('total')}   "
        f"ETA {m.group('eta').strip()}"
    )
    return desc, progress


class Splash:
    """Frameless centered window with a status label. Updated from the worker
    thread via .set_status(); polls a stop predicate via .run_until()."""

    def __init__(self, title="Loading...", initial="Starting up..."):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)        # frameless
        self.root.attributes("-topmost", True)

        w, h = 560, 160
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg="#141414")

        # Accent border (a 2px frame around the dark inner)
        border = tk.Frame(self.root, bg="#00c850")
        border.pack(fill="both", expand=True, padx=0, pady=0)
        inner = tk.Frame(border, bg="#141414")
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(
            inner, text="Korean OCR Translator",
            fg="#ffffff", bg="#141414",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(20, 4))

        self._status_var = tk.StringVar(value=initial)
        tk.Label(
            inner, textvariable=self._status_var,
            fg="#cccccc", bg="#141414",
            font=("Consolas", 10),
            wraplength=520, justify="center",
        ).pack(pady=(8, 6))

        self._progress_var = tk.StringVar(value="")
        tk.Label(
            inner, textvariable=self._progress_var,
            fg="#00c850", bg="#141414",
            font=("Consolas", 9, "bold"),
        ).pack(pady=(0, 14))

        self.root.update_idletasks()

    def set_status(self, text):
        """Thread-safe-ish status update — schedules on the tk main loop.
        If `text` looks like a tqdm progress line, the bar is parsed into a
        clean status + progress pair instead of being dumped raw."""
        status, progress = _format_line(text)
        try:
            self.root.after(0, lambda: self._status_var.set(status))
            self.root.after(0, lambda: self._progress_var.set(progress))
        except Exception:
            pass

    def run_until(self, done_event, poll_ms=100):
        """Pump tk events until done_event is set, then close.
        Call from the thread that constructed Splash (main thread)."""
        def _check():
            if done_event.is_set():
                try:
                    self.root.destroy()
                except Exception:
                    pass
                return
            self.root.after(poll_ms, _check)
        self.root.after(poll_ms, _check)
        try:
            self.root.mainloop()
        except Exception:
            pass

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def run_with_splash(work_fn, title="Loading...", initial="Starting up..."):
    """Show a splash, run `work_fn(set_status)` on a worker thread, close
    splash when it returns. Returns (result, exception_or_None)."""
    splash = Splash(title=title, initial=initial)
    done = threading.Event()
    box = {"result": None, "exc": None}

    def worker():
        try:
            box["result"] = work_fn(splash.set_status)
        except BaseException as e:
            box["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    splash.run_until(done)
    t.join(timeout=0.5)
    return box["result"], box["exc"]
