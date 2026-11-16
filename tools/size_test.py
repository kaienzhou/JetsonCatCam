"""How big must the cat be in the picture for the detector to find it?

    python3 tools/size_test.py room.jpg ~/jetson-inference/data/images/cat_1.jpg [more cats ...]
    python3 tools/size_test.py --engine models/yolov8n_384x512.fp16.engine room.jpg cat_1.jpg ...

Pastes each cat photo into the background image (e.g. a frame from your own camera)
at several sizes and prints the best 'cat' confidence for each size. Sizes are the
width of the pasted photo as a fraction of the frame width. Without --engine the
jetson-inference ssd-mobilenet-v2 is tested, with --engine a YOLOv8 TensorRT engine.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import jetson.utils as utils
except ImportError:
    import jetson_utils as utils

FRACTIONS = [0.60, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08]


class Ssd(object):
    def __init__(self):
        try:
            import jetson.inference as inference
        except ImportError:
            import jetson_inference as inference
        self.net = inference.detectNet("ssd-mobilenet-v2", threshold=0.05)

    def best_cat(self, bgr):
        img = utils.cudaFromNumpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        found = [d.Confidence for d in self.net.Detect(img, overlay="none")
                 if self.net.GetClassDesc(d.ClassID) == "cat"]
        return max(found or [0.0])


class Yolo(object):
    def __init__(self, engine):
        from catcam.yolo import YoloTrt
        self.det = YoloTrt(engine, utils, conf=0.05, iou=0.5)  # cats only

    def best_cat(self, bgr):
        return max([s for name, s, _ in self.det.detect(bgr) if name == "cat"] or [0.0])


def paste(background, cat, fraction):
    h, w = background.shape[:2]
    cw = max(8, int(w * fraction))
    ch = max(8, int(cat.shape[0] * cw / float(cat.shape[1])))
    if ch > h:  # photo taller than the frame at this width: skip this size
        return None
    small = cv2.resize(cat, (cw, ch), interpolation=cv2.INTER_AREA)
    out = background.copy()
    x = (w - cw) // 2
    y = min(h - ch, int(h * 0.55))  # lower part of the frame, like a cat on the floor
    out[y:y + ch, x:x + cw] = small
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", help="YOLOv8 TensorRT engine; default is ssd-mobilenet-v2")
    ap.add_argument("background")
    ap.add_argument("cats", nargs="+")
    args = ap.parse_args()

    detector = Yolo(args.engine) if args.engine else Ssd()
    background = cv2.imread(args.background)
    print("%-12s" % "cat photo" + "".join("%7d%%" % round(f * 100) for f in FRACTIONS), flush=True)
    for path in args.cats:
        cat = cv2.imread(path)
        row = ["%-12s" % path.split("/")[-1]]
        for f in FRACTIONS:
            pasted = paste(background, cat, f)
            row.append("%8s" % "-" if pasted is None else "%8.2f" % detector.best_cat(pasted))
        print("".join(row), flush=True)


if __name__ == "__main__":
    main()
