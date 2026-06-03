# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the on-screen Korean OCR translator.
# Build with:    pyinstaller OnscreenTranslator.spec --noconfirm --clean
# Output:        dist/OnscreenTranslator/  (zip this folder for sharing)

from PyInstaller.utils.hooks import collect_all, collect_submodules


def _gather(pkg, **kw):
    """collect_all returns (datas, binaries, hiddenimports). Merge them."""
    d, b, h = collect_all(pkg, **kw)
    return d, b, h


datas, binaries, hiddenimports = [], [], []

# Heavy ML packages — let PyInstaller pull every submodule + native dll.
for pkg in (
    "torch",
    "torchvision",
    "bitsandbytes",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "easyocr",
    "PyQt6",
    "cv2",
    "dxcam",
    "scipy",
    "skimage",
    "shapely",
    "pyclipper",
    "ninja",
):
    try:
        d, b, h = _gather(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] warning: collect_all({pkg!r}) failed: {e}")

# Hidden imports that PyInstaller occasionally misses.
hiddenimports += collect_submodules("bitsandbytes")
hiddenimports += collect_submodules("transformers.models.qwen2")
hiddenimports += [
    "PyQt6.sip",
    "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
    "comtypes.stream",
]

block_cipher = None


a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # We don't use any of these — shaves a few hundred MB.
        "matplotlib", "tkinter.test", "test", "unittest",
        "pytest", "IPython", "jupyter", "notebook",
        "PyQt5", "PySide6", "PySide2",
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OnscreenTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX often corrupts torch/cv2 dlls
    console=False,             # windowed app, no console flash
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="icon.ico",         # add an .ico beside the spec to brand the exe
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OnscreenTranslator",
)
