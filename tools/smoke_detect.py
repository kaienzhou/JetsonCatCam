"""Smoke test on a Jetson: run a detector on image files and print detections and speed.

    python3 tools/smoke_detect.py ~/jetson-inference/data/images/cat_0.jpg [more images ...]
    python3 tools/smoke_detect.py --engine models/yolov8n_384x512.fp16.engine cat_0.jpg ...

Without --engine this is jetson-inference's ssd-mobilenet-v2 (its TensorRT engine is
built on the first run, several minutes); with --engine a YOLOv8 TensorRT engine
(see tools/export_yolo.sh) that reports all COCO classes.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import jetson.utils as utils
except ImportError:
    import jetson_utils as utils


def ssd():
    try:
        import jetson.inference as inference
    except ImportError:
        import jetson_inference as inference
    net = inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    def run(path):
        img = utils.loadImage(path)
        dets = net.Detect(img, overlay="none")
        return [(net.GetClassDesc(d.ClassID), d.Confidence) for d in dets]
    return run


def yolo(engine):
    from catcam.yolo import YoloTrt
    det = YoloTrt(engine, utils, conf=0.4, iou=0.5, class_ids=None)

    def run(path):
        return [(name, score) for name, score, _ in det.detect(cv2.imread(path))]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", help="YOLOv8 TensorRT engine; default is ssd-mobilenet-v2")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    t0 = time.time()
    run = yolo(args.engine) if args.engine else ssd()
    print("detector ready in %.1f s" % (time.time() - t0))
    for path in args.images:
        run(path)  # warm-up: the first call includes CUDA setup
        n = 10
        t = time.time()
        for _ in range(n):
            found = run(path)
        ms = (time.time() - t) / n * 1000
        text = ", ".join("%s %.2f" % d for d in found)
        print("%s: %d detections, %.1f ms/image (incl. file decode): %s" % (
            os.path.basename(path), len(found), ms, text or "-"))


if __name__ == "__main__":
    main()
