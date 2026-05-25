import threading
import queue
import collections
import logging
import time
import cv2
import torch
import ocr as ocr_backend
from translate import translate, warmup
from stream import Stream


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


MONITOR_INDEX = 1
TARGET_FPS = 60
OCR_HEIGHT = 720
X_GAP_RATIO = 0.8
Y_GAP_RATIO = 0.4
IOU_CARRY_THRESHOLD = 0.5
MIN_BOX_W = 16
MIN_BOX_H = 10
OCR_UPSCALE = 2.0


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
    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            stop_event.wait(0.01)
            continue

        fh, fw = frame.shape[:2]
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
                if OCR_UPSCALE and OCR_UPSCALE != 1.0:
                    crop = cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
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


def main():
    global latest_frame

    setup_logging()

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}"
          + (f" ({torch.cuda.get_device_name(0)})" if cuda_ok else ""))

    print("[main] pre-warming translation model ...", flush=True)
    warm_t0 = _t()
    warmup()
    print(f"[main] warmup done in {_ms(warm_t0, _t())/1000:.1f}s", flush=True)

    detector_thread = threading.Thread(target=detector_worker, daemon=True)
    detector_thread.start()
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    cv2.namedWindow("screen-ocr", cv2.WINDOW_NORMAL)

    stream = Stream(monitor=MONITOR_INDEX, target_fps=TARGET_FPS,
                    crop_top_ratio=0.1, crop_bottom_ratio=0.1)
    stream.start()

    main_last_report = _t()
    main_acc = {"get": 0.0, "draw": 0.0, "show": 0.0, "n": 0}

    try:
        while True:
            t0 = _t()
            frame = stream.get_frame()
            t_get = _t()

            with latest_frame_lock:
                latest_frame = frame

            display = frame.copy()

            with results_lock:
                snapshot = list(results)

            for box, text, trans in snapshot:
                (pt1, pt2) = box
                cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)
                label = trans or text
                if label:
                    cv2.putText(display, label, (pt1[0], max(pt1[1] - 6, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                cv2.LINE_AA)
            t_draw = _t()

            cv2.imshow("screen-ocr", display)
            key = cv2.waitKey(1) & 0xFF
            t_show = _t()

            main_acc["get"] += _ms(t0, t_get)
            main_acc["draw"] += _ms(t_get, t_draw)
            main_acc["show"] += _ms(t_draw, t_show)
            main_acc["n"] += 1
            if t_show - main_last_report >= 2.0:
                n = max(1, main_acc["n"])
                print(f"[main] last {n} frames avg: "
                      f"get={main_acc['get']/n:5.1f}ms "
                      f"draw={main_acc['draw']/n:5.1f}ms "
                      f"show+wait={main_acc['show']/n:5.1f}ms "
                      f"boxes={len(snapshot):2d}",
                      flush=True)
                main_acc = {"get": 0.0, "draw": 0.0, "show": 0.0, "n": 0}
                main_last_report = t_show

            if key == ord('q'):
                break
    finally:
        stream.stop()
        stop_event.set()
        detector_thread.join(timeout=1.0)
        ocr_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
