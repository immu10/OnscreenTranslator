import threading
import cv2
import dxcam
import torch
import easyocr

MONITOR_INDEX = 0
TARGET_FPS = 60
HASH_HAMMING_MAX = 12
HASH_HAMMING_MAX_DORMANT = 8
TRACK_MAX_CENTROID_DIST = 150
TRACK_TTL_FRAMES = 600
DORMANT_AFTER_FRAMES = 2
OCR_HEIGHT = 720
X_GAP_RATIO = 0.8
Y_GAP_RATIO = 0.4


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

cuda_ok = torch.cuda.is_available()
print(f"CUDA available: {cuda_ok}"
      + (f" ({torch.cuda.get_device_name(0)})" if cuda_ok else ""))
reader = easyocr.Reader(['en'], gpu=cuda_ok)

latest_frame = None
latest_frame_lock = threading.Lock()
latest_tracks = []
detections_lock = threading.Lock()
stop_event = threading.Event()

tracks = {}
next_track_id = 1
frame_counter = 0


def ahash(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def update_tracks(frame, boxes):
    global next_track_id, frame_counter
    frame_counter += 1
    fh, fw = frame.shape[:2]

    observations = []
    for (x1, y1), (x2, y2) in boxes:
        mx = int((x2 - x1) * 0.1)
        my = int((y2 - y1) * 0.1)
        x1c = max(0, x1 - mx)
        y1c = max(0, y1 - my)
        x2c = min(fw, x2 + mx)
        y2c = min(fh, y2 + my)
        if x2c - x1c < 4 or y2c - y1c < 4:
            continue
        crop = frame[y1c:y2c, x1c:x2c]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        observations.append({
            "box": ((x1, y1), (x2, y2)),
            "hash": ahash(crop),
            "centroid": (cx, cy),
        })

    used_obs, used_tid = set(), set()
    assignments = {}

    visible_cands = []
    for oi, obs in enumerate(observations):
        for tid, tr in tracks.items():
            if frame_counter - tr["last_seen"] > DORMANT_AFTER_FRAMES:
                continue
            dx = obs["centroid"][0] - tr["centroid"][0]
            dy = obs["centroid"][1] - tr["centroid"][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > TRACK_MAX_CENTROID_DIST:
                continue
            ham = hamming(obs["hash"], tr["hash"])
            if ham > HASH_HAMMING_MAX:
                continue
            visible_cands.append((ham, dist, oi, tid))

    visible_cands.sort()
    for ham, dist, oi, tid in visible_cands:
        if oi in used_obs or tid in used_tid:
            continue
        assignments[oi] = tid
        used_obs.add(oi)
        used_tid.add(tid)

    dormant_cands = []
    for oi, obs in enumerate(observations):
        if oi in used_obs:
            continue
        for tid, tr in tracks.items():
            if tid in used_tid:
                continue
            if frame_counter - tr["last_seen"] <= DORMANT_AFTER_FRAMES:
                continue
            ham = hamming(obs["hash"], tr["hash"])
            if ham > HASH_HAMMING_MAX_DORMANT:
                continue
            dormant_cands.append((ham, oi, tid))

    dormant_cands.sort()
    for ham, oi, tid in dormant_cands:
        if oi in used_obs or tid in used_tid:
            continue
        assignments[oi] = tid
        used_obs.add(oi)
        used_tid.add(tid)

    new_tracks = {}
    for oi, obs in enumerate(observations):
        tid = assignments.get(oi)
        if tid is None:
            tid = next_track_id
            next_track_id += 1
        new_tracks[tid] = {
            "box": obs["box"],
            "hash": obs["hash"],
            "centroid": obs["centroid"],
            "last_seen": frame_counter,
        }

    for tid, tr in tracks.items():
        if tid in new_tracks:
            continue
        if frame_counter - tr["last_seen"] <= TRACK_TTL_FRAMES:
            new_tracks[tid] = tr

    tracks.clear()
    tracks.update(new_tracks)
    return [(tid, tr["box"]) for tid, tr in tracks.items()
            if tr["last_seen"] == frame_counter]


def ocr_worker():
    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            stop_event.wait(0.01)
            continue
        h, w = frame.shape[:2]
        scale = OCR_HEIGHT / h
        small = cv2.resize(frame, (int(w * scale), OCR_HEIGHT))
        horizontal_list, free_list = reader.detect(small)
        inv = 1.0 / scale
        boxes = []
        for x_min, x_max, y_min, y_max in horizontal_list[0]:
            boxes.append((
                (int(x_min * inv), int(y_min * inv)),
                (int(x_max * inv), int(y_max * inv)),
            ))
        for quad in free_list[0]:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            boxes.append((
                (int(min(xs) * inv), int(min(ys) * inv)),
                (int(max(xs) * inv), int(max(ys) * inv)),
            ))
        grouped = group_boxes(boxes)
        active = update_tracks(frame, grouped)
        with detections_lock:
            latest_tracks[:] = active


worker = threading.Thread(target=ocr_worker, daemon=True)
worker.start()

cv2.namedWindow("screen-ocr", cv2.WINDOW_NORMAL)

camera = dxcam.create(output_idx=MONITOR_INDEX, output_color="BGR")
camera.start(target_fps=TARGET_FPS, video_mode=True)

try:
    while True:
        frame = camera.get_latest_frame()

        with latest_frame_lock:
            latest_frame = frame

        with detections_lock:
            active = list(latest_tracks)

        for tid, (pt1, pt2) in active:
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"#{tid}", (pt1[0], max(pt1[1] - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("screen-ocr", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.stop()
    stop_event.set()
    worker.join(timeout=1.0)
    cv2.destroyAllWindows()
