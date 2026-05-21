import threading
from paddleocr import PaddleOCR

LANG = "korean"

_lock = threading.Lock()
_ocr = None


def _ensure():
    global _ocr
    if _ocr is None:
        print("[ocr] loading PaddleOCR lang={} ...".format(LANG), flush=True)
        _ocr = PaddleOCR(
            lang=LANG,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("[ocr] PaddleOCR ready", flush=True)


def _poly_to_box(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return (
        (int(min(xs)), int(min(ys))),
        (int(max(xs)), int(max(ys))),
    )


def _field(res, key):
    if res is None:
        return []
    try:
        v = res[key]
        if v is not None:
            return v
    except (TypeError, KeyError):
        pass
    return getattr(res, key, []) or []


def _predict(image):
    with _lock:
        _ensure()
        return _ocr.predict(image)


def detect(image):
    boxes = []
    results = _predict(image) or []
    for res in results:
        for poly in _field(res, "dt_polys"):
            boxes.append(_poly_to_box(poly))
    return boxes


def recognize(crop):
    parts = []
    results = _predict(crop) or []
    for res in results:
        for t in _field(res, "rec_texts"):
            if isinstance(t, str) and t:
                parts.append(t)
    return " ".join(parts).strip()
