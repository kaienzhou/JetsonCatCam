"""How much does a colour cast (like a NoIR camera in daylight) hurt cat detection?

    python3 tools/tint_test.py ~/jetson-inference/data/images/cat_0.jpg [more images ...]

For every image prints the best 'cat' confidence for the original, for the image with a
simulated cast, and for the cast image after gray-world white balance. Two casts are
tried: "live" (measured on a real NoIR frame indoors in daylight) and "strong".
"""
import sys

import cv2
import numpy as np

try:
    import jetson.inference as inference
    import jetson.utils as utils
except ImportError:
    import jetson_inference as inference
    import jetson_utils as utils

# per-channel multipliers in BGR order
CASTS = {"live": (1.20, 0.76, 1.03), "strong": (1.35, 0.60, 1.05)}


def apply_cast(bgr, gains):
    return np.clip(bgr.astype(np.float32) * np.array(gains, np.float32), 0, 255).astype(np.uint8)


def gray_world(bgr):
    means = bgr.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1.0)
    return np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def best_cat(net, bgr):
    img = utils.cudaFromNumpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    best, top = 0.0, "-"
    found = net.Detect(img, overlay="none")
    for d in found:
        if net.GetClassDesc(d.ClassID) == "cat":
            best = max(best, d.Confidence)
    if found:
        d = max(found, key=lambda d: d.Confidence)
        top = "%s %.2f" % (net.GetClassDesc(d.ClassID), d.Confidence)
    return best, top


def main(paths):
    net = inference.detectNet("ssd-mobilenet-v2", threshold=0.05)
    print("%-12s %-9s %-26s %-26s" % ("image", "original", "cast -> as is", "cast -> gray-world"), flush=True)
    totals = {}
    for path in paths:
        bgr = cv2.imread(path)
        row = ["%-12s" % path.split("/")[-1]]
        orig, _ = best_cat(net, bgr)
        row.append("%-9.2f" % orig)
        totals.setdefault("original", []).append(orig)
        for name, gains in sorted(CASTS.items()):
            tinted = apply_cast(bgr, gains)
            raw, top_raw = best_cat(net, tinted)
            fixed, top_fix = best_cat(net, gray_world(tinted))
            row.append("%s %.2f (top %s)" % (name, raw, top_raw))
            row.append("%.2f" % fixed)
            totals.setdefault(name + " as is", []).append(raw)
            totals.setdefault(name + " gray-world", []).append(fixed)
        print("  ".join(row), flush=True)
    print("MEAN best-cat confidence:", ", ".join(
        "%s=%.2f" % (k, sum(v) / len(v)) for k, v in sorted(totals.items())), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
