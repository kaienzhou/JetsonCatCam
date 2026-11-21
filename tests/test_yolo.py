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


@unittest.skipIf(np is None, "numpy not installed")
class WindowTest(unittest.TestCase):
    def test_pixel_window_cuts_whole_pixels_and_reports_what_it_cut(self):
        (x0, y0, x1, y1), real = yolo.pixel_window((0.25, 0.25, 0.75, 0.75), 640, 480)
        self.assertEqual((x0, y0, x1, y1), (160, 120, 480, 360))
        self.assertEqual(real, (0.25, 0.25, 0.75, 0.75))
        _, real = yolo.pixel_window((0.1234, 0.1, 0.5, 0.5), 640, 480)  # 78.98 px -> 79 px
        self.assertAlmostEqual(real[0], 79 / 640.0)

    def test_pixel_window_is_clipped_and_tiny_ones_are_refused(self):
        (x0, y0, x1, y1), _ = yolo.pixel_window((-0.2, 0.5, 1.3, 1.2), 640, 480)
        self.assertEqual((x0, y0, x1, y1), (0, 240, 640, 480))
        self.assertIsNone(yolo.pixel_window((0.5, 0.5, 0.51, 0.9), 640, 480))

    def test_boxes_in_a_window_map_back_to_the_whole_picture(self):
        got = yolo.from_window((0.5, 0.5, 1.0, 1.0), (0.25, 0.25, 0.75, 0.75))
        for g, w in zip(got, (0.5, 0.5, 0.75, 0.75)):
            self.assertAlmostEqual(g, w)


class FakeContext(object):
    """Stands in for the TensorRT execution context: 'finds' one cat in the middle of the input."""

    def __init__(self, det):
        self.det = det

    def execute_v2(self, bindings):
        self.det.view_out[:] = 0
        self.det.view_out[0:4, 0] = (256, 192, 100, 80)   # cx, cy, w, h in input pixels
        self.det.view_out[4 + yolo.CAT, 0] = 0.9
        return True


@unittest.skipIf(np is None, "numpy not installed")
class DetectWindowTest(unittest.TestCase):
    def make(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        det = yolo.YoloTrt.__new__(yolo.YoloTrt)
        det.cv2, det.conf, det.iou, det.class_ids, det.names = cv2, 0.3, 0.5, [yolo.CAT], yolo.COCO
        det.h, det.w = 384, 512
        det.view_in = np.zeros((3, 384, 512), np.float32)
        det.view_out = np.zeros((84, 8), np.float32)
        det.bindings = []
        det.ctx = FakeContext(det)
        return det

    def test_without_a_window_the_whole_frame_is_used(self):
        det = self.make()
        (name, score, box), = det.detect(np.zeros((960, 1280, 3), np.uint8))
        self.assertEqual(name, "cat")
        for g, w in zip(box, (206 / 512.0, 152 / 384.0, 306 / 512.0, 232 / 384.0)):
            self.assertAlmostEqual(g, w, places=4)

    def test_a_window_is_what_the_network_sees_and_boxes_come_back_in_frame_coordinates(self):
        det = self.make()
        frame = np.zeros((960, 1280, 3), np.uint8)
        frame[240:720, 640:1280] = 200                     # the window (0.5, 0.25, 1.0, 0.75)
        (name, score, box), = det.detect(frame, (0.5, 0.25, 1.0, 0.75))
        self.assertAlmostEqual(float(det.view_in.mean()), 200 / 255.0, places=2)  # nothing else in the input
        want = (0.5 + 206 / 512.0 * 0.5, 0.25 + 152 / 384.0 * 0.5,
                0.5 + 306 / 512.0 * 0.5, 0.25 + 232 / 384.0 * 0.5)
        for g, w in zip(box, want):
            self.assertAlmostEqual(g, w, places=4)

    def test_a_window_that_is_too_small_falls_back_to_the_whole_frame(self):
        det = self.make()
        (name, score, box), = det.detect(np.zeros((960, 1280, 3), np.uint8), (0.5, 0.5, 0.505, 0.505))
        self.assertAlmostEqual(box[0], 206 / 512.0, places=4)


if __name__ == "__main__":
    unittest.main()
