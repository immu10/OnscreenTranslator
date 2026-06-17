# OnscreenTranslator

Real-time Korean-to-English screen translator. Captures your monitor, detects text
on-screen via OCR, translates it with a local LLM, and overlays the English
output beside the source text — all offline, no cloud APIs.

Built for reading Korean wuxia/manhwa webnovels alongside the page rather than
swapping tabs to a separate translator.

> ⚠️ **Do not run this with kernel-mode anti-cheat games** (Riot Vanguard /
> Valorant / League, EAC, BattlEye, FACEIT, etc.). The combination of
> screen capture via `dxcam`, a topmost overlay drawing over other windows,
> and `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` — which hides the
> overlay from screen-capture APIs — matches exactly the behavioral pattern
> these anti-cheats flag as cheating overlays. Best case: the anti-cheat
> refuses to launch the game. Worst case: hardware-ID ban. Use this on
> manhwa, VNs, YouTube, single-player games, or anything *without* kernel
> anti-cheat. **You assume all risk if you ignore this.**

## Contents

- [Overview](#overview)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [Building an .exe](#building-an-exe)
- [Performance notes](#performance-notes)
- [Known limitations](#known-limitations)
- [Architecture choices worth knowing](#architecture-choices-worth-knowing)

## Overview

Captures your chosen monitor region, runs Korean OCR on each frame's text
boxes, translates each new phrase locally with a 4-bit-quantized Qwen 2.5
7B chat model (no cloud API), and floats the English result beside the
source text in a transparent click-through overlay. Built for reading
Korean wuxia/manhwa webnovels alongside the page rather than swapping
tabs to a separate translator.

Designed to be:

- **Fully offline** after the one-time model download — no API keys, no
  network calls per translation.
- **Portable** — settings, model cache, OCR weights, and logs all live in
  the app folder. Nothing in `%APPDATA%`, nothing in the registry, nothing
  in `C:\`. Unzip anywhere user-writable and run.
- **Distributable as a single .exe** (PyInstaller onedir) to friends with
  NVIDIA GPUs — no Python install on the recipient.
- **Pausable** — double-click the floating handle to halt detection,
  translation, and the overlay; double-click again to resume.

## Tech stack

| Layer | Tool | Why this choice |
|-------|------|-----------------|
| Screen capture | `dxcam` (DXGI Desktop Duplication) | GPU-backed, lowest-latency Windows path; supports per-region capture |
| OCR detect + recognize | `easyocr` (PyTorch) | Solid Korean accuracy, single-package install |
| LLM translation | `Qwen2.5-7B-Instruct` + `bitsandbytes` nf4 4-bit | Strong Korean→English; ~5 GB VRAM |
| Inference runtime | `transformers` + `torch` (CUDA 12.1) | Standard HF stack, `AutoModelForCausalLM` API |
| Overlay UI | `PyQt6` transparent click-through `QWidget` | Native widgets, per-pixel alpha, click-through via `Qt.WindowTransparentForInput` |
| Floating handle + tray | `PyQt6` (`QSystemTrayIcon`, custom `QWidget`) | Discord-overlay-style draggable button, plus tray fallback |
| Splash window | `tkinter` (stdlib) | GDI-only — doesn't init Qt's GPU plugin before bnb loads |
| Capture exclusion | Win32 `SetWindowDisplayAffinity` (`WDA_EXCLUDEFROMCAPTURE`) | Prevents dxcam from re-OCR'ing the overlay's own translations |
| Settings persistence | JSON beside the app | Portable, human-editable |
| Live log viewer | Custom stdout/stderr tee → ring buffer → Qt `QPlainTextEdit` | See model load + OCR/translate timings without a console |
| Packaging | PyInstaller (onedir) | Single folder you can zip and share; no Python on the recipient |

Tested on Windows 11 + Python 3.11 + NVIDIA RTX 4070 SUPER. AMD/Intel GPUs
are not supported in this build (`bitsandbytes` is CUDA-only on Windows);
swapping translation to llama.cpp + Vulkan would unlock cross-vendor
support — see [Performance notes](#performance-notes).

## Architecture

Four threads share state through three lock-guarded objects. Capture is
"latest-frame wins"; detection is gated by a thumbnail diff to skip static
screens; OCR runs on a drop-oldest queue so the translator never falls
behind on stale crops.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                  DXGI Desktop Duplication                       │
   │                         (dxcam)                                 │
   └─────────────────────────────┬───────────────────────────────────┘
                                 │ raw frames (target 60 FPS)
                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                   capture_worker  [thread #1]                   │
   │                  pulls frames, overwrites state                 │
   └─────────────────────────────┬───────────────────────────────────┘
                                 │ writes
                                 ▼
                  ╔═══════════════════════════════╗
                  ║      latest_frame  (lock)     ║   shared state
                  ╚═══════════════╤═══════════════╝
                                  │ snapshot + 32×32 thumb-diff
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                   detector_worker  [thread #2]                  │
   │   • same-frame skip (abs-diff < 5 → idle 50ms)                  │
   │   • EasyOCR.detect() on 720p downscale                          │
   │   • union-find box merge (group_boxes)                          │
   │   • IoU carry-over of (text, translation) from prev frame       │
   │   • honours paused_event → idle + clears results                │
   └─────────────────────┬────────────────────────────┬──────────────┘
                         │ new crops only             │ writes
                         ▼                            ▼
            ╔══════════════════════════╗   ╔══════════════════════════╗
            ║   ocr_queue (drop-old)   ║   ║       results  (lock)    ║◄──┐
            ║   bounded depth = 4      ║   ║  list[(box, ko, en)]     ║   │
            ╚════════════╤═════════════╝   ╚══════════════════════════╝   │
                         │                                                │
                         ▼                                                │
   ┌─────────────────────────────────────────────────────────────────┐    │
   │                     ocr_worker  [thread #3]                     │    │
   │   • cv2.resize cubic upscale (ocr_upscale)                      │    │
   │   • EasyOCR.recognize() on the crop                             │    │
   │   • translation_cache hit? → reuse                              │    │
   │   • else → translate() through Qwen 7B (bnb nf4)                │    │
   │   • IoU back-match → patch the right entry in results ──────────┼────┘
   └────────────────────┬───────────────────────┬────────────────────┘
                        │                       │
                        ▼                       ▼
            ╔══════════════════════════╗   ┌──────────────────────┐
            ║   translation_cache      ║   │   Qwen 2.5 7B 4-bit  │
            ║      ko → en  (dict)     ║   │   bnb nf4 / CUDA     │
            ╚══════════════════════════╝   └──────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                   Qt main thread  [thread #4]                   │
   │   • Overlay paints results @ 30 Hz (reads results under lock)   │
   │   • FloatingButton: drag, double-click pause, menu              │
   │   • System tray icon                                            │
   │   • Settings / Logs dialogs (all WDA_EXCLUDEFROMCAPTURE)        │
   └─────────────────────────────────────────────────────────────────┘
```

Per-stage detail:

- **Capture** — `dxcam` Desktop Duplication grabs a configurable region of a
  chosen monitor (top/bottom percent crop or a drag-picked rectangle).
- **Detection** — EasyOCR on a 720p downscale to find text bounding boxes; boxes
  are merged via union-find (custom `group_boxes`) to coalesce paragraph-level
  blocks. IoU ≥ 0.5 against the previous frame carries existing translations
  forward so the overlay stays stable across small camera jitter.
- **Recognition** — EasyOCR's Korean recognizer runs on the full-res crop of
  each new box (optional 2× cubic upscale for stylized fonts).
- **Translation** — Qwen2.5-7B-Instruct in 4-bit (bitsandbytes nf4) on local
  CUDA. Content-keyed cache eliminates re-translation of repeated phrases.
- **Display** — PyQt6 transparent click-through overlay floats English text
  beside each box (auto-positions right/left/below/above to stay on-screen),
  excluded from screen capture via `SetWindowDisplayAffinity` so it isn't
  re-OCR'd by dxcam.
- **Pause** — `paused_event` short-circuits the detector loop and clears
  `results` under the lock; the OCR worker idles naturally as the queue
  drains. Resume forces one detection pass by resetting `prev_thumb`.

## Project layout

| Path | Role |
|------|------|
| [main.py](main.py) | Orchestration: threads, shared state, `main()`, GPU check |
| [paths.py](paths.py) | Portable path helpers (exe folder when frozen, project root in dev) |
| [screen/stream.py](screen/stream.py) | `Stream` class wrapping dxcam with region cropping |
| [text/ocr.py](text/ocr.py) | EasyOCR backend: `detect()` and `recognize()` |
| [text/translate.py](text/translate.py) | Qwen 4-bit loading + `translate()` + `warmup()` |
| [ui/overlay.py](ui/overlay.py) | PyQt6 click-through overlay + floating handle + tray |
| [ui/settings.py](ui/settings.py) | Persistent settings dialog + region picker |
| [ui/logs.py](ui/logs.py) | stdout/stderr ring buffer + live log viewer |
| [ui/splash.py](ui/splash.py) | tkinter splash shown during model warmup |
| [run/run.ps1](run/run.ps1) / [run/run.sh](run/run.sh) | Wrappers that timestamp output to `logs/` |
| [OnscreenTranslator.spec](OnscreenTranslator.spec) | PyInstaller spec for the .exe build |

## Installation

Recreate the conda env from scratch (the install order matters because EasyOCR
pulls `opencv-python-headless` which collides with `opencv-python` on disk).

```powershell
conda deactivate
conda env remove -n ocr -y
conda create -n ocr python=3.11 -y
conda activate ocr

pip install -r requirements.txt
pip uninstall opencv-python-headless -y
pip install opencv-python
```

See [requirements.txt](requirements.txt) for the pinned versions and the
header comment for the rationale.

## Running

```powershell
.\run\run.ps1   # PowerShell — also pipes all output into logs/run-<timestamp>.log
# or
python main.py  # plain
```

On first run, downloads to the app folder:
- Qwen2.5-7B-Instruct safetensors (~5 GB) into `.hf_cache/`
- EasyOCR Korean recognizer (~80 MB) into `.easyocr/`

A tkinter splash window shows live status (tqdm progress bars are parsed
and surfaced as `desc / pct% / cur / total / ETA`). Subsequent starts skip
the download and warm up in ~30 s, then the floating green "T" handle
appears.

**To quit:** right-click the green floating handle (or the tray icon) → Quit.
Ctrl+C in the terminal also exits cleanly.

## Configuration

Most knobs live in `settings.json` (auto-created next to the app) and can be
edited from the in-app **Settings...** dialog. Live-applied settings take
effect on Save without restart; restart-required ones reapply by re-running
the capture stream in place.

| Setting | Live? | What it does |
|--------|------|--------------|
| `monitor_index` | restart capture | Which display to capture (`output_idx`) |
| `custom_region` | restart capture | `[x1,y1,x2,y2]` monitor-local; overrides crop ratios |
| `crop_top_ratio` / `crop_bottom_ratio` | restart capture | Fallback strip-crop % if no custom region |
| `box_color` / `bg_color` / `text_color` | live | RGBA for outline / text background / text |
| `show_box_outline` | live | Toggle the green outline rectangle |
| `font_family` / `font_size_min` / `font_size_max` | live | Translation label font |
| `ocr_upscale` | live | Per-crop cubic upscale before recognition (helps stylized fonts) |

Engine constants (queue depth, IoU threshold, gap ratios for box merging) are
still at the top of [main.py](main.py) if you need to tune them.

## Building an .exe

Produces a portable folder you can zip and hand to a friend. Tested on
Windows 11 + Python 3.11 + NVIDIA RTX-class GPU.

### One-time setup

```powershell
# Use the same venv you run the app from, then:
pip install pyinstaller
```

### Build

```powershell
pyinstaller OnscreenTranslator.spec --noconfirm --clean
```

Output: `dist/OnscreenTranslator/` — contains `OnscreenTranslator.exe` plus
all its DLLs and Python runtime. First build takes ~10 minutes; the bundle
is ~2–3 GB.

### Share

1. Zip `dist/OnscreenTranslator/` into `OnscreenTranslator.zip` (~1.2 GB).
2. Send to a friend.
3. They unzip **anywhere user-writable** (Documents, Desktop) — NOT into
   `Program Files`. The app writes settings, model cache, and OCR models
   next to the exe.
4. Double-click `OnscreenTranslator.exe`.
5. SmartScreen will warn — they click "More info" → "Run anyway".
6. On first launch, the splash window appears and the Qwen weights (~5 GB)
   download to `.hf_cache/` next to the exe. Takes 10+ minutes depending
   on connection.
7. Subsequent launches: ~30 s warmup, then the floating green "T" handle
   appears top-right. Right-click it for Settings / Logs / Quit.

### Recipient requirements

- Windows 11
- NVIDIA GPU with ≥8 GB VRAM (currently same class as developer's GPU (4070S))
- NVIDIA driver R535 or newer (`nvidia-smi` to check)
- ~7 GB free disk space (model + cache)

If their GPU/driver doesn't qualify, the app shows a MessageBox and exits
cleanly instead of crashing.

### Troubleshooting

**"VCRUNTIME140.dll missing"** → install the Microsoft Visual C++ 2015–2022
Redistributable (x64). Most Win11 machines already have it.

**Splash shows but never finishes** → check `boxID.log` next to the exe,
or flip `console=True` in the spec and rebuild to see stdout in a console
window.

**SmartScreen blocks every time** → sign the exe (`signtool sign /a /fd
SHA256 OnscreenTranslator.exe`) using a code-signing cert. Not free.

**Anti-virus quarantines it** → expected for an unsigned binary that
captures the screen and uses `SetWindowDisplayAffinity`. Whitelist the
folder or sign it.

## Performance notes

On an RTX 4070 SUPER, current pipeline:

- Detection: ~100 ms/frame at 720p input
- Recognition: ~70 ms/crop (2× upscaled)
- Translation: ~2 s per new phrase (Qwen 7B + bnb 4-bit @ ~3 tok/s)
- Cache hits: ~free (dict lookup)

Translation dominates wall-clock by far. Speedups not yet applied:

- **llama.cpp + Vulkan** — same model in GGUF Q4_K_M, ~15-25 tok/s on
  NVIDIA, also unlocks AMD/Intel GPUs. Larger refactor of `text/translate.py`.
- **AWQ** for Qwen — pre-quantized, ~3× faster kernels than bnb. Blocked
  on `autoawq` failing to install on Windows without VS Build Tools.
- **Batched translation** when multiple new boxes arrive at once.

## Known limitations

- Two-frame visual flicker between the moment a box is detected and the moment
  its translation lands; same-frame check eliminates this on static pages.
- OCR confuses some Korean particles in stylized manhwa fonts (e.g. `을` → `올`,
  spaces dropped). PaddleOCR's Korean recognizer is better at this but its
  CUDA stack conflicts with torch on Windows.
- Click-through is via Qt's `WindowTransparentForInput`. On older Windows
  builds this may need a Win32 `WS_EX_TRANSPARENT` fallback.
- 12 GB+ VRAM recommended (Qwen 7B 4-bit ~5 GB + EasyOCR ~1 GB + headroom).
- **Anti-cheat incompatibility.** dxcam + topmost overlay + capture-exclusion
  flag will trip Riot Vanguard, EAC, etc. Use it on single-player or non-game
  content (manhwa, VNs, YouTube) — not on competitive multiplayer titles.

## Architecture choices worth knowing

- **Content-keyed translation cache, not ID-based tracking.** Earlier
  iterations tried to maintain stable IDs across frames using perceptual
  hashes; collisions on stylized text made this unreliable. Now identity is
  the recognized string itself.
- **PyQt6 imported lazily inside `main()`.** Importing it at module level
  before bitsandbytes finishes loading Qwen can cause silent CUDA conflicts on
  Windows. (`ui.logs` and `ui.settings` are imported earlier — those are
  module-level Qt imports, not QApplication construction, which is the actual
  trigger.)
- **Drop-oldest queue between detector and OCR worker.** Bounded at 4; when
  full, the *oldest* crop is dropped, not the newest. Keeps OCR working on
  current content rather than ancient screenshots.
- **Pre-warm the LLM before starting threads.** Otherwise the first OCR job
  blocks behind a ~10 s model load, queue backlog explodes, and translations
  appear ~15 s stale.
- **Splash uses tkinter, not Qt.** Showing a Qt splash before warmup would
  initialize Qt's graphics plugin, which collides with bitsandbytes on Windows.
  tkinter is pure GDI and stays out of the way.
- **Overlay excluded from screen capture** via `SetWindowDisplayAffinity`
  (`WDA_EXCLUDEFROMCAPTURE`). Without this, the overlay's own translations
  get re-captured by dxcam and recursively OCR'd into garbage.
- **Portable app folder.** Settings, HF cache, and EasyOCR models all live
  next to the exe (or in the project root during dev). No `%APPDATA%`, no
  registry, nothing in `C:\` — the whole app is a self-contained directory.

## Tags

`LLM` · `Qt` · `Qwen` · `OCR`
