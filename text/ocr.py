import threading
import torch
import easyocr

from paths import easyocr_model_dir

LANGS = ['ko', 'en']

_lock = threading.Lock()
_reader = None


def _ensure():
    global _reader
    if _reader is None:
        gpu = torch.cuda.is_available()
        model_dir = easyocr_model_dir()
        print(f"[ocr] loading EasyOCR langs={LANGS} gpu={gpu} "
              f"model_dir={model_dir} ...", flush=True)
        _reader = easyocr.Reader(
            LANGS, gpu=gpu,
            model_storage_directory=model_dir,
            user_network_directory=model_dir,
        )
        print("[ocr] EasyOCR ready", flush=True)


def detect(image):
    with _lock:
        _ensure()
        horizontal_list, free_list = _reader.detect(image)
    boxes = []
    for x_min, x_max, y_min, y_max in horizontal_list[0]:
        boxes.append((
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
        ))
    for quad in free_list[0]:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        boxes.append((
            (int(min(xs)), int(min(ys))),
            (int(max(xs)), int(max(ys))),
        ))
    return boxes


def recognize(crop):
    with _lock:
        _ensure()
        results = _reader.readtext(crop, detail=0, paragraph=True)
    if not results:
        return ""
    return " ".join(r for r in results if isinstance(r, str) and r).strip()
