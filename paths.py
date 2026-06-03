"""Where the app reads/writes runtime state.

Portable layout: everything (settings, model cache, OCR models) lives inside
the app folder. In dev that's the project directory; when frozen by
PyInstaller that's the folder containing the exe. Recipient must therefore
unzip the exe somewhere user-writable (Documents, Desktop, etc.) — not
Program Files.
"""

import os
import sys


def is_frozen():
    """True when running from a PyInstaller-built exe."""
    return getattr(sys, "frozen", False)


def app_dir():
    """Folder containing the exe (frozen) or the project root (dev)."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def settings_file():
    return os.path.join(app_dir(), "settings.json")


def hf_cache_dir():
    d = os.path.join(app_dir(), ".hf_cache")
    os.makedirs(d, exist_ok=True)
    return d


def easyocr_model_dir():
    d = os.path.join(app_dir(), ".easyocr")
    os.makedirs(d, exist_ok=True)
    return d


def bundle_path(*parts):
    """Path to a read-only bundled asset.
    - Frozen onedir: relative to the exe folder.
    - Frozen onefile: relative to PyInstaller's _MEIPASS extraction dir.
    - Source: relative to the project root.
    """
    if is_frozen():
        base = getattr(sys, "_MEIPASS", app_dir())
    else:
        base = app_dir()
    return os.path.join(base, *parts)
