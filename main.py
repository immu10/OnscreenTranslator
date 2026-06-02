import threading
import queue
import collections
import logging
import time
import cv2
import torch
import ocr as ocr_backend
import settings
from translate import translate, warmup
from stream import Stream
# ui (PyQt6) is imported lazily inside main() AFTER model warmup to avoid
# Qt's graphics plugin grabbing GPU context before bitsandbytes finishes
# loading Qwen — that conflict silently crashes bnb on Windows.


def _t():
    return time.perf_counter()


def _ms(t0, t1):
    return (t1 - t0) * 1000


class DropOldestQueue:
    """Bounded queue that drops the OLDEST item when full instead of blocking
    or rejecting the newest. Keeps the worker working on fresh data."""

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._dq = collections.deque()
        self._cond = threading.Condition()
        self.dropped_total = 0

    def put(self, item):
        with self._cond:
            if len(self._dq) >= self.maxsize:
                self._dq.popleft()
                self.dropped_total += 1
            self._dq.append(item)
            self._cond.notify()

    def get(self, timeout=None):
        with self._cond:
            if not self._dq:
                self._cond.wait(timeout)
            if not self._dq:
                raise queue.Empty
            return self._dq.popleft()

    def qsize(self):
        with self._cond:
            return len(self._dq)


TARGET_FPS = 60
OCR_HEIGHT = 720
X_GAP_RATIO = 0.8
Y_GAP_RATIO = 0.4
IOU_CARRY_THRESHOLD = 0.5
MIN_BOX_W = 16
MIN_BOX_H = 10
# OCR_UPSCALE / MONITOR_INDEX / crop ratios are now in settings.SETTINGS.


def group_boxes(boxes):
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    pads = []
    for (_, y1), (_, y2) in boxes:
        h = max(1, y2 - y1)
        pads.append((h * X_GAP_RATIO, h * Y_GAP_RATIO))

    for i in range(n):
        (ax1, ay1), (ax2, ay2) = boxes[i]
        pxi, pyi = pads[i]
        for j in range(i + 1, n):
            (bx1, by1), (bx2, by2) = boxes[j]
            pxj, pyj = pads[j]
            px = max(pxi, pxj)
            py = max(pyi, pyj)
            if (ax1 - px <= bx2 and bx1 - px <= ax2
                    and ay1 - py <= by2 and by1 - py <= ay2):
                union(i, j)

    groups = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)

    merged = []
    for group in groups.values():
        xs1 = [b[0][0] for b in group]
        ys1 = [b[0][1] for b in group]
        xs2 = [b[1][0] for b in group]
        ys2 = [b[1][1] for b in group]
        merged.append(((min(xs1), min(ys1)), (max(xs2), max(ys2))))
    return merged


def iou(a, b):
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


# Shared state — populated by main(), read by the worker threads.
latest_frame = None
latest_frame_lock = threading.Lock()
stop_event = threading.Event()

results = []
results_lock = threading.Lock()

translation_cache = {}
translation_cache_lock = threading.Lock()

ocr_queue = DropOldestQueue(maxsize=4)

box_log = None  # initialized in main()


def detector_worker():
    prev_thumb = None
    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            stop_event.wait(0.01)
            continue

        fh, fw = frame.shape[:2]

        # Same-frame check: if the screen hasn't meaningfully changed since the
        # last iteration, skip detection entirely and leave results untouched.
        # Cheap thumbnail diff (32x32 grayscale, max-pixel-diff threshold).
        thumb = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY)
        if prev_thumb is not None:
            if int(cv2.absdiff(thumb, prev_thumb).max()) < 5:
                stop_event.wait(0.05)  # static frame, don't burn the GPU
                continue
        prev_thumb = thumb

        scale = OCR_HEIGHT / fh
        small = cv2.resize(frame, (int(fw * scale), OCR_HEIGHT))

        small_boxes = ocr_backend.detect(small)

        inv = 1.0 / scale
        raw = []
        for (x1, y1), (x2, y2) in small_boxes:
            raw.append((
                (int(x1 * inv), int(y1 * inv)),
                (int(x2 * inv), int(y2 * inv)),
            ))
        grouped = group_boxes(raw)

        with results_lock:
            prev = list(results)

        new_results = []
        for box in grouped:
            (x1, y1), (x2, y2) = box
            if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                continue

            text, trans = "", ""
            best = 0.0
            for (pb, pt, ptr) in prev:
                io = iou(box, pb)
                if io > best:
                    best = io
                    if io >= IOU_CARRY_THRESHOLD and pt:
                        text, trans = pt, ptr

            if not text:
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(fw, x2)
                y2c = min(fh, y2)
                if x2c - x1c >= 4 and y2c - y1c >= 4:
                    crop = frame[y1c:y2c, x1c:x2c].copy()
                    ocr_queue.put((box, crop, _t()))

            new_results.append((box, text, trans))

        with results_lock:
            results[:] = new_results

        for box, text, trans in new_results:
            (x1, y1), (x2, y2) = box
            box_log.info("det box=(%d,%d,%d,%d) text=%r", x1, y1, x2, y2, text)


def ocr_worker():
    while not stop_event.is_set():
        try:
            box, crop, queued_at = ocr_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        t_pop = _t()
        try:
            try:
                upscale = settings.SETTINGS.get("ocr_upscale", 1.0)
                if upscale and upscale != 1.0:
                    crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                                      interpolation=cv2.INTER_CUBIC)
                text = ocr_backend.recognize(crop)
            except Exception:
                box_log.exception("ocr recognize failed")
                text = ""
            t_recog = _t()
            if not text:
                print(f"[ocr] wait={_ms(queued_at,t_pop):6.1f}ms "
                      f"recog={_ms(t_pop,t_recog):6.1f}ms text='' (skipped)",
                      flush=True)
                continue

            cached = False
            with translation_cache_lock:
                trans = translation_cache.get(text)
                if trans is not None:
                    cached = True
            if not cached:
                try:
                    trans = translate(text)
                except Exception:
                    box_log.exception("translate failed for text=%r", text)
                    trans = ""
                with translation_cache_lock:
                    translation_cache[text] = trans
            t_trans = _t()

            with results_lock:
                best = 0.0
                best_i = -1
                for i, (pb, _, _) in enumerate(results):
                    io = iou(box, pb)
                    if io > best:
                        best = io
                        best_i = i
                if best_i >= 0 and best >= IOU_CARRY_THRESHOLD:
                    results[best_i] = (results[best_i][0], text, trans)
                    matched = True
                else:
                    matched = False
            t_end = _t()

            print(f"[ocr] wait={_ms(queued_at,t_pop):6.1f}ms "
                  f"recog={_ms(t_pop,t_recog):6.1f}ms "
                  f"trans={_ms(t_recog,t_trans):7.1f}ms"
                  f"{' (cache)' if cached else '       '} "
                  f"update={_ms(t_trans,t_end):5.1f}ms "
                  f"e2e={_ms(queued_at,t_end):7.1f}ms "
                  f"matched={matched} text={text!r}",
                  flush=True)

            box_log.info("ocr %s text=%r trans=%r",
                         "cache" if cached else "new", text, trans)
        except Exception:
            box_log.exception("ocr_worker loop error")


def setup_logging():
    global box_log
    logging.basicConfig(
        filename="boxID.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                  "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    box_log = logging.getLogger("boxID")


def capture_worker(stream):
    """Continuously pulls frames from the screen stream and publishes the
    latest one to `latest_frame` for the detector to consume."""
    global latest_frame
    while not stop_event.is_set():
        try:
            frame = stream.get_frame()
        except Exception as e:
            print(f"[capture] get_frame error: {type(e).__name__}: {e}",
                  flush=True)
            stop_event.wait(0.1)
            continue
        if frame is None:
            stop_event.wait(0.005)
            continue
        with latest_frame_lock:
            latest_frame = frame


def get_results_snapshot():
    """Thread-safe accessor for the UI's QTimer to pull current results."""
    with results_lock:
        return list(results)


def main():
    setup_logging()
    settings.load()

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}"
          + (f" ({torch.cuda.get_device_name(0)})" if cuda_ok else ""))

    print("[main] pre-warming translation model ...", flush=True)
    warm_t0 = _t()
    warmup()
    print(f"[main] warmup done in {_ms(warm_t0, _t())/1000:.1f}s", flush=True)

    # Import PyQt6 only AFTER the LLM is fully loaded in VRAM, otherwise
    # Qt's graphics plugin init can crash bitsandbytes silently on Windows.
    import ui

    monitor_index = settings.SETTINGS["monitor_index"]
    stream = Stream(
        monitor=monitor_index,
        target_fps=TARGET_FPS,
        crop_top_ratio=settings.SETTINGS["crop_top_ratio"],
        crop_bottom_ratio=settings.SETTINGS["crop_bottom_ratio"],
        custom_region=settings.SETTINGS.get("custom_region"),
    )
    stream.start()

    capture_thread = threading.Thread(
        target=capture_worker, args=(stream,), daemon=True
    )
    capture_thread.start()
    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    detector_thread.start()
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    try:
        # Qt event loop runs on the main thread; blocks until window closes
        # or stop_event is set by another thread.
        def restart_capture(new_monitor, new_top, new_bot,
                            new_custom_region=...):
            """Called from the Qt thread when settings change capture params.
            Stops the current dxcam stream and starts a new one in place.
            Workers keep running; capture_worker sees None briefly during the
            swap and resumes once the new stream produces frames.

            Pass new_custom_region=... (Ellipsis) to leave custom_region as-is;
            pass None explicitly to clear it; pass a 4-tuple to set it."""
            try:
                print(f"[main] reconfiguring capture: monitor={new_monitor} "
                      f"top={new_top} bottom={new_bot} "
                      f"region={new_custom_region}", flush=True)
                new_region = stream.reconfigure(
                    monitor=new_monitor,
                    crop_top_ratio=new_top,
                    crop_bottom_ratio=new_bot,
                    custom_region=new_custom_region,
                )
                return new_region
            except Exception as e:
                print(f"[main] restart_capture failed: {e}", flush=True)
                return None

        ui.run(
            monitor_index=monitor_index,
            region=stream.region,
            results_getter=get_results_snapshot,
            stop_event=stop_event,
            restart_capture=restart_capture,
        )
    finally:
        stop_event.set()
        stream.stop()
        capture_thread.join(timeout=1.0)
        detector_thread.join(timeout=1.0)
        ocr_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
