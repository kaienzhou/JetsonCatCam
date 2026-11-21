"""How far away can the camera + detector still find a cat?

    python3 tools/range_test.py room.jpg ~/jetson-inference/data/images/cat_*.jpg \
        --pipe 640x480:models/yolov8n_384x512.fp16.engine \
        --pipe 1024x768:models/yolov8n_768x1024.fp16.engine

Cuts each cat out of its photo, pastes it into a picture of your room at the size a real
cat has at 2 .. 7 m, and runs the whole chain the camera frames go through: sensor
picture (1640x1232 by default) -> scaled to the frame size the detector gets -> network.
Every --pipe is "<frame width>x<frame height>:<engine>" and is tested the same way, so
you can compare setups (frame size, engine size) without moving a cat around.

"<frame width>x<frame height>/<filter>:<engine>" picks how the sensor picture is shrunk:
area (default, what nvvidconv's interpolation-method=5 comes close to), linear, or
nearest (nvvidconv's default, which drops pixels).

"win@<scale>:<engine>" is a pipe that looks at a window of the sensor picture around the
cat instead of the whole picture (scale = network pixels per sensor pixel, 1 = no
shrinking). It shows what zooming in on a zone could gain.

The table shows the best 'cat' confidence (mean over all photos and places) and, in
brackets, in how many of the tries it was at least --min-conf, which is what the
tracker needs. It is optimistic: photos are sharper than a real cat seen through a
small lens (--degrade adds some blur and noise), so look at the differences between
the rows more than at the absolute numbers.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


PLACES = [(0.30, 0.62), (0.50, 0.70), (0.72, 0.55)]  # where the cat's centre goes (x, y of the picture)


def cut_out(path, det):
    """The cat in a photo: a tight crop around the biggest 'cat' box, or the whole photo."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("cannot read %s" % path)
    best = None
    for name, score, box in det.detect(img):
        if name == "cat" and (best is None or score > best[0]):
            best = (score, box)
    if best is None:
        return img
    h, w = img.shape[:2]
    l, t, r, b = best[1]
    mx, my = 0.04 * (r - l), 0.04 * (b - t)
    x1, x2 = int(max(0, (l - mx)) * w), int(min(1, (r + mx)) * w)
    y1, y2 = int(max(0, (t - my)) * h), int(min(1, (b + my)) * h)
    return img[y1:y2, x1:x2] if x2 - x1 > 8 and y2 - y1 > 8 else img


def paste(background, cat, fraction, place):
    """-> (background with the cat crop `fraction` of the picture width wide, soft edges;
    centre of the cat in pixels)."""
    h, w = background.shape[:2]
    cw = max(8, int(round(w * fraction)))
    ch = max(8, int(round(cat.shape[0] * cw / float(cat.shape[1]))))
    small = cv2.resize(cat, (cw, ch), interpolation=cv2.INTER_AREA)
    out = background.copy()
    x = int(min(max(0, place[0] * w - cw / 2.0), w - cw))
    y = int(min(max(0, place[1] * h - ch / 2.0), h - ch))
    alpha = np.zeros((ch, cw), np.float32)
    edge = max(1, min(cw, ch) // 12)
    alpha[edge:ch - edge, edge:cw - edge] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, edge / 2.5))[:, :, None]
    roi = out[y:y + ch, x:x + cw].astype(np.float32)
    out[y:y + ch, x:x + cw] = (alpha * small + (1.0 - alpha) * roi).astype(np.uint8)
    return out, (x + cw // 2, y + ch // 2)


def window(img, centre, size, scale):
    """The part of `img` around `centre` that fills a network input of `size` (w, h) at `scale`."""
    h, w = img.shape[:2]
    ww, wh = min(w, int(round(size[0] / scale))), min(h, int(round(size[1] / scale)))
    x = int(min(max(0, centre[0] - ww / 2.0), w - ww))
    y = int(min(max(0, centre[1] - wh / 2.0), h - wh))
    crop = img[y:y + wh, x:x + ww]
    if (ww, wh) == tuple(size):
        return crop
    return cv2.resize(crop, tuple(size), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def degrade(img, rng):
    """Roughly what a small lens and an indoor sensor add: softness and noise."""
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    noise = rng.normal(0.0, 4.0, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def best_cat(det, frame):
    return max([s for name, s, _ in det.detect(frame) if name == "cat"] or [0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("background", help="picture of the room, ideally a frame of your own camera")
    ap.add_argument("cats", nargs="+", help="photos of cats")
    ap.add_argument("--pipe", action="append", required=True, metavar="WxH:ENGINE",
                    help="frame size given to the detector and the TensorRT engine; repeat to compare")
    ap.add_argument("--sensor", default="1640x1232", help="size of the camera picture (default 1640x1232)")
    ap.add_argument("--fov", type=float, default=62.2, help="horizontal field of view in degrees (Pi camera V2: 62.2)")
    ap.add_argument("--cat-width", type=float, default=0.35, help="width of the cat in metres (default 0.35)")
    ap.add_argument("--distances", default="2,3,4,5,6,7", help="metres, comma separated")
    ap.add_argument("--min-conf", type=float, default=0.3, help="the threshold the tracker uses (config min_conf)")
    ap.add_argument("--degrade", action="store_true", help="blur and noise like a real lens and sensor")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import jetson.utils as utils
    except ImportError:
        import jetson_utils as utils
    from catcam.yolo import YoloTrt

    sw, sh = [int(v) for v in args.sensor.lower().split("x")]
    bg = cv2.imread(args.background)
    if bg is None:
        raise SystemExit("cannot read %s" % args.background)
    bg = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_CUBIC)
    distances = [float(v) for v in args.distances.split(",")]
    width_at = lambda d: 2.0 * d * math.tan(math.radians(args.fov / 2.0))
    fractions = [args.cat_width / width_at(d) for d in distances]
    rng = np.random.RandomState(1)

    filters = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
    pipes = []  # (label, frame size or None for a window, window scale or shrink filter, engine)
    for spec in args.pipe:
        kind, _, engine = spec.partition(":")
        name = os.path.basename(engine).replace("yolov8n_", "").replace(".fp16.engine", "")
        if kind.startswith("win@"):
            pipes.append(("window x%s %s" % (kind[4:], name), None, float(kind[4:]), engine))
        else:
            size, _, how = kind.partition("/")
            fw, fh = [int(v) for v in size.lower().split("x")]
            if how and how not in filters:
                raise SystemExit("filter must be one of %s" % ", ".join(sorted(filters)))
            label = "%dx%d%s %s" % (fw, fh, "/" + how if how else "", name)
            pipes.append((label, (fw, fh), filters[how or "area"], engine))

    # the photos are cut out once, with the first engine
    det = YoloTrt(pipes[0][3], utils, conf=0.2, iou=args.iou)
    crops = [(os.path.basename(p), cut_out(p, det)) for p in args.cats]
    del det

    print("cat %.0f cm wide, camera %.1f deg wide, %s -> detector" % (args.cat_width * 100, args.fov, args.sensor))
    print("%-34s" % "distance (m)" + "".join("%12.1f" % d for d in distances))
    print("%-34s" % "cat width in the picture" + "".join("%11.1f%%" % (f * 100) for f in fractions))
    scenes = []  # per distance: [(picture, cat centre)]
    for f in fractions:
        row = []
        for _, crop in crops:
            for place in PLACES:
                scene, centre = paste(bg, crop, f, place)
                row.append((degrade(scene, rng) if args.degrade else scene, centre))
        scenes.append(row)
    empty = [degrade(bg, rng) if args.degrade else bg]

    for label, size, how, engine in pipes:
        det = YoloTrt(engine, utils, conf=0.02, iou=args.iou)

        def frame_of(scene, centre):
            if size is None:  # a window; `how` is its scale
                return window(scene, centre, (det.w, det.h), how)
            return scene if size == (sw, sh) else cv2.resize(scene, size, interpolation=how)

        cells = []
        for row in scenes:
            scores = [best_cat(det, frame_of(scene, centre)) for scene, centre in row]
            hit = sum(1 for s in scores if s >= args.min_conf) / float(len(scores))
            cells.append("%5.2f (%3d%%)" % (sum(scores) / len(scores), round(hit * 100)))
        # nothing pasted: should stay low
        nothing = max(best_cat(det, frame_of(empty[0], (int(sw * x), int(sh * y)))) for x, y in PLACES)
        print("%-34s" % label + "".join("%12s" % c for c in cells) + "   (empty room: %.2f)" % nothing)
        sys.stdout.flush()
        del det


if __name__ == "__main__":
    main()
