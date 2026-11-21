"""Draw the web app's icons (catcam/static/*.png). Needs OpenCV + numpy; run it from the project folder:

    python tools/make_icons.py

The pictures are in the repository, so you only need this when you change the design.
"""
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catcam", "static")
BG = (39, 25, 15)           # BGR: dark blue-grey, the app's background
FUR = (36, 166, 245)        # amber
INNER = (60, 100, 200)      # inside of the ears
EYE = (30, 20, 12)
NOSE = (160, 130, 240)


def draw(size):
    big = size * 4  # draw large, shrink for smooth edges
    img = np.empty((big, big, 3), np.uint8)
    img[:] = BG

    def p(x, y):
        return int(x * big), int(y * big)

    def tri(points, color):
        cv2.fillPoly(img, [np.array([p(*q) for q in points], np.int32)], color, cv2.LINE_AA)

    # everything stays inside the middle 60%: Android may crop the icon to a circle
    tri([(0.27, 0.47), (0.29, 0.21), (0.47, 0.35)], FUR)
    tri([(0.73, 0.47), (0.71, 0.21), (0.53, 0.35)], FUR)
    tri([(0.31, 0.43), (0.32, 0.28), (0.42, 0.36)], INNER)
    tri([(0.69, 0.43), (0.68, 0.28), (0.58, 0.36)], INNER)
    cv2.ellipse(img, p(0.5, 0.57), (int(0.25 * big), int(0.22 * big)), 0, 0, 360, FUR, -1, cv2.LINE_AA)
    for x in (0.40, 0.60):
        cv2.ellipse(img, p(x, 0.55), (int(0.032 * big), int(0.048 * big)), 0, 0, 360, EYE, -1, cv2.LINE_AA)
        cv2.circle(img, p(x + 0.009, 0.535), int(0.011 * big), (255, 255, 255), -1, cv2.LINE_AA)
    tri([(0.47, 0.635), (0.53, 0.635), (0.5, 0.67)], NOSE)
    for side in (-1, 1):
        for dy in (-0.012, 0.012):
            cv2.line(img, p(0.5 + side * 0.14, 0.655 + dy), p(0.5 + side * 0.30, 0.64 + dy * 4), (235, 240, 245),
                     max(2, big // 170), cv2.LINE_AA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    for name, size in (("icon-512.png", 512), ("icon-192.png", 192), ("apple-touch-icon.png", 180)):
        path = os.path.abspath(os.path.join(OUT, name))
        cv2.imwrite(path, draw(size))
        print("wrote", path, os.path.getsize(path), "bytes")
