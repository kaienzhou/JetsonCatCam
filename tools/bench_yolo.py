"""How fast is a YOLO engine on this board, and where does the time go?

    python3 tools/bench_yolo.py models/yolov8n_384x512.fp16.engine --frame 640x480
    python3 tools/bench_yolo.py models/yolov8n_768x1024.fp16.engine --frame 1024x768

Times one detect() call split into its parts (frame -> network input, the network itself,
reading the result) on a noise picture of the given frame size, and prints the memory the
process used for the engine. Stop the catcam service first (or accept noisy numbers): the
GPU is shared.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import jetson.utils as utils
except ImportError:
    import jetson_utils as utils

from catcam.yolo import YoloTrt, decode, letterbox_params


def meminfo_mb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--frame", default="640x480", help="size of the camera frame, WxH")
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    fw, fh = [int(v) for v in args.frame.lower().split("x")]

    # the GPU shares the board's memory: what the engine takes shows in "available", not in RSS
    avail, rss = meminfo_mb("MemAvailable"), rss_mb()
    det = YoloTrt(args.engine, utils, conf=0.3, iou=0.5)
    cv2 = det.cv2
    frame = np.random.RandomState(0).randint(0, 255, (fh, fw, 3)).astype(np.uint8)
    for _ in range(5):
        det.detect(frame)
    print("engine input %dx%d, frame %dx%d, memory taken: %.0f MB of the board's (RSS +%.0f MB)"
          % (det.w, det.h, fw, fh, avail - meminfo_mb("MemAvailable"), rss_mb() - rss))

    pre = gpu = post = 0.0
    scale, nw, nh, dx, dy = letterbox_params(fw, fh, det.w, det.h)
    for _ in range(args.runs):
        t0 = time.time()
        img = frame if (nw, nh) == (fw, fh) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if (nw, nh) != (det.w, det.h):
            canvas = np.full((det.h, det.w, 3), 114, np.uint8)
            canvas[dy:dy + nh, dx:dx + nw] = img
            img = canvas
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (det.w, det.h), swapRB=True, crop=False)
        np.copyto(det.view_in, blob.reshape(3, det.h, det.w))
        t1 = time.time()
        det.ctx.execute_v2(det.bindings)
        t2 = time.time()
        decode(det.view_out, det.conf, det.iou, det.class_ids)
        t3 = time.time()
        pre, gpu, post = pre + t1 - t0, gpu + t2 - t1, post + t3 - t2
    n = float(args.runs)
    print("prepare %.1f ms   network %.1f ms   decode %.1f ms   total %.1f ms (%.1f per second)"
          % (pre / n * 1000, gpu / n * 1000, post / n * 1000, (pre + gpu + post) / n * 1000,
             n / (pre + gpu + post)))


if __name__ == "__main__":
    main()
