"""Pure-numpy parts of the YOLO detector: NMS, decoding, letterbox maths."""
import unittest

try:
    import numpy as np
except ImportError:  # e.g. the system python in WSL; the Jetson and the export venv have numpy
    np = None

if np is not None:
    from catcam import yolo


def head(rows):
    """Build a (4 + 80, N) YOLOv8 output from [(cx, cy, w, h, {class_id: score}), ...]."""
    out = np.zeros((4 + 80, len(rows)), np.float32)
    for i, (cx, cy, w, h, scores) in enumerate(rows):
        out[0:4, i] = (cx, cy, w, h)
        for cid, s in scores.items():
            out[4 + cid, i] = s
    return out


@unittest.skipIf(np is None, "numpy not installed")
class NmsTest(unittest.TestCase):
    def test_overlapping_boxes_keep_only_the_best(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.6, 0.9, 0.7], np.float32)
        self.assertEqual(yolo.nms(boxes, scores, 0.5), [1, 2])

    def test_single_box(self):
        self.assertEqual(yolo.nms(np.array([[0, 0, 5, 5]], np.float32), np.array([0.5], np.float32), 0.5), [0])


@unittest.skipIf(np is None, "numpy not installed")
class DecodeTest(unittest.TestCase):
    def test_cat_box_is_converted_from_centre_size_to_corners(self):
        out = head([(100, 80, 40, 20, {yolo.CAT: 0.9})])
        (cid, score, box), = yolo.decode(out, conf=0.4, iou=0.5)
        self.assertEqual(cid, yolo.CAT)
        self.assertAlmostEqual(score, 0.9, places=5)
        self.assertEqual(box, (80.0, 70.0, 120.0, 90.0))

    def test_below_threshold_is_dropped(self):
        self.assertEqual(yolo.decode(head([(10, 10, 5, 5, {yolo.CAT: 0.3})]), conf=0.4), [])

    def test_duplicates_of_one_class_are_merged_but_other_classes_survive(self):
        out = head([(100, 100, 50, 50, {yolo.CAT: 0.9}),
                    (102, 101, 50, 50, {yolo.CAT: 0.8}),           # same cat, shifted a little
                    (100, 100, 50, 50, {yolo.COCO.index("dog"): 0.7})])  # a dog on the same spot
        found = yolo.decode(out, conf=0.4, iou=0.5)
        self.assertEqual([(c, round(s, 2)) for c, s, _ in found],
                         [(yolo.CAT, 0.9), (yolo.COCO.index("dog"), 0.7)])

    def test_class_filter_ignores_other_classes(self):
        out = head([(10, 10, 5, 5, {yolo.COCO.index("dog"): 0.99}),
                    (60, 60, 5, 5, {yolo.CAT: 0.5})])
        found = yolo.decode(out, conf=0.4, class_ids=[yolo.CAT])
        self.assertEqual([c for c, _, _ in found], [yolo.CAT])

    def test_results_are_sorted_best_first(self):
        out = head([(10, 10, 5, 5, {yolo.CAT: 0.5}), (100, 100, 5, 5, {yolo.CAT: 0.8})])
        self.assertEqual([round(s, 1) for _, s, _ in yolo.decode(out)], [0.8, 0.5])


@unittest.skipIf(np is None, "numpy not installed")
class LetterboxTest(unittest.TestCase):
    def test_same_aspect_needs_no_padding(self):
        self.assertEqual(yolo.letterbox_params(640, 480, 512, 384), (0.8, 512, 384, 0, 0))

    def test_wide_frame_is_padded_top_and_bottom(self):
        scale, nw, nh, dx, dy = yolo.letterbox_params(1280, 720, 512, 384)
        self.assertAlmostEqual(scale, 0.4)
        self.assertEqual((nw, nh, dx, dy), (512, 288, 0, 48))

    def test_box_maps_back_to_normalized_frame_coordinates(self):
        # 1280x720 frame in a 512x384 input: the image occupies y = 48 .. 336
        scale, _, _, dx, dy = yolo.letterbox_params(1280, 720, 512, 384)
        box = yolo.to_frame((256 - 51.2, 192 - 28.8, 256 + 51.2, 192 + 28.8), 1280, 720, scale, dx, dy)
        for got, want in zip(box, (0.4, 0.4, 0.6, 0.6)):
            self.assertAlmostEqual(got, want, places=5)

    def test_boxes_are_clipped_to_the_frame(self):
        self.assertEqual(yolo.to_frame((-10, -10, 600, 500), 640, 480, 0.8, 0, 0), (0.0, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
