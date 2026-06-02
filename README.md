# OnscreenTranslator

Real-time Korean-to-English screen translator. Captures your monitor, detects text
on-screen via OCR, translates it with a local LLM, and overlays the English
output beside the source text — all offline, no cloud APIs.

Built for reading Korean wuxia/manhwa webnovels alongside the page rather than
swapping tabs to a separate translator.

## Pipeline

```
dxcam ─► capture thread ─► detector thread ─► OCR/translate thread ─► PyQt6 overlay
            (60 FPS)         (EasyOCR detect    (EasyOCR recognize       (transparent
                              on downscaled       + Qwen 7B 4-bit         click-through
                              frame)              translate)              window beside
                                                                          each box)
```

- **Capture** — `dxcam` Desktop Duplication grabs a configurable region of a
  chosen monitor (top/bottom percent crop to skip browser chrome).
- **Detection** — EasyOCR on a 720p downscale to find text bounding boxes; boxes
  are merged via union-find (custom `group_boxes`) to coalesce paragraph-level
  blocks.
- **Recognition** — EasyOCR's Korean recognizer runs on the full-res crop of
  each new box (optional 2× cubic upscale for stylized fonts).
- **Translation** — Qwen2.5-7B-Instruct in 4-bit (bitsandbytes nf4) on local
  CUDA. Content-keyed cache eliminates re-translation of repeated phrases.
- **Display** — PyQt6 transparent click-through overlay floats English text
  beside each box (auto-positions right/left/below/above to stay on-screen).

## Files

| File | Role |
|------|------|
| [main.py](main.py) | Orchestration: threads, shared state, `main()` |
| [stream.py](stream.py) | `Stream` class wrapping dxcam with region cropping |
| [ocr.py](ocr.py) | EasyOCR backend: `detect()` and `recognize()` |
| [translate.py](translate.py) | Qwen 4-bit loading + `translate()` + `warmup()` |
| [ui.py](ui.py) | PyQt6 click-through overlay + system tray exit |
| [run.ps1](run.ps1) / [run.sh](run.sh) | Wrappers that timestamp output to `logs/` |

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

Optional: set `KMP_DUPLICATE_LIB_OK=TRUE` if you ever mix Paddle in (Intel
OpenMP duplicate-init warning).

## Running

```powershell
.\run.ps1       # PowerShell — also pipes all output into logs/run-<timestamp>.log
# or
python main.py  # plain
```

On first run, downloads:
- Qwen2.5-7B-Instruct safetensors (~15 GB) into `.hf_cache/`
- EasyOCR Korean recognizer (~80 MB) into `~/.EasyOCR/`

Subsequent starts are ~5–10 s warmup, then the overlay appears.

**To quit:** right-click the green tray icon → Quit. (No taskbar window; the
overlay is a Tool-class always-on-top window.)

## Configuration knobs

Top of `main.py`:

| Constant | Default | What it does |
|----------|---------|--------------|
| `MONITOR_INDEX` | 1 | Which display to capture (dxcam's `output_idx`) |
| `TARGET_FPS` | 60 | dxcam capture rate |
| `OCR_HEIGHT` | 720 | Detection runs on a frame resized to this height |
| `OCR_UPSCALE` | 2.0 | Per-crop upscale before recognition |
| `MIN_BOX_W` / `MIN_BOX_H` | 16 / 10 | Drop boxes smaller than this (UI noise filter) |
| `IOU_CARRY_THRESHOLD` | 0.5 | IoU needed to carry forward a translation between frames |
| `X_GAP_RATIO` / `Y_GAP_RATIO` | 0.8 / 0.4 | How aggressively to merge adjacent boxes into one |

The dxcam capture is cropped to skip the top and bottom 10% of the monitor
(`crop_top_ratio` / `crop_bottom_ratio` on `Stream(...)`) — adjust if your
browser chrome / taskbar takes more than that.

## Performance notes

On an RTX 4070 SUPER, current pipeline:

- Detection: ~100 ms/frame at 720p input
- Recognition: ~70 ms/crop (2× upscaled)
- Translation: ~2 s per new phrase (Qwen 7B + bnb 4-bit @ ~3 tok/s)
- Cache hits: ~free (dict lookup)

Translation dominates wall-clock by far. Speedups not yet applied:

- **AWQ** for Qwen — pre-quantized, ~3× faster kernels. Blocked on `autoawq`
  failing to install on Windows without VS Build Tools.
- **Process isolation** for the LLM, freeing GPU contention with EasyOCR.
- **Batched translation** when multiple new boxes arrive at once.

## Known limitations

- Two-frame visual flicker between the moment a box is detected and the moment
  its translation lands; same-frame check eliminates this on static pages.
- OCR confuses some Korean particles in stylized manhwa fonts (e.g. `을` → `올`,
  spaces dropped). PaddleOCR's Korean recognizer is better at this but its
  CUDA stack conflicts with torch on Windows.
- Click-through is via Qt's `WindowTransparentForInput`. On older Windows
  builds this may need a Win32 `WS_EX_TRANSPARENT` fallback.
- Single monitor / single capture region per run; no live reconfig.
- 12 GB+ VRAM recommended (Qwen 7B 4-bit ~5 GB + EasyOCR ~1 GB + headroom).

## Architecture choices worth knowing

- **Content-keyed translation cache, not ID-based tracking.** Earlier
  iterations tried to maintain stable IDs across frames using perceptual
  hashes; collisions on stylized text made this unreliable. Now identity is
  the recognized string itself.
- **PyQt6 imported lazily inside `main()`.** Importing it at module level
  before bitsandbytes finishes loading Qwen can cause silent CUDA conflicts on
  Windows.
- **Drop-oldest queue between detector and OCR worker.** Bounded at 4; when
  full, the *oldest* crop is dropped, not the newest. Keeps OCR working on
  current content rather than ancient screenshots.
- **Pre-warm the LLM before starting threads.** Otherwise the first OCR job
  blocks behind a ~10 s model load, queue backlog explodes, and translations
  appear ~15 s stale.
