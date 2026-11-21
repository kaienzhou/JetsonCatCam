"""The picture maths of tools/range_test.py: it is only useful if the cat really is the size it claims."""
import importlib.util
import os
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "range_test.py")


def load():
    spec = importlib.util.spec_from_file_location("range_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(cv2 is not None and np is not None, "needs OpenCV and numpy")
class PasteTest(unittest.TestCase):
    def setUp(self):
        self.tool = load()
        self.bg = np.zeros((100, 200, 3), np.uint8)
        self.cat = np.full((20, 40, 3), 255, np.uint8)

    def test_the_cat_is_the_asked_fraction_of_the_picture_wide(self):
        out, (cx, cy) = self.tool.paste(self.bg, self.cat, 0.25, (0.5, 0.5))
        columns = np.nonzero(out[cy, :, 0] > 128)[0]       # a row through the cat's middle
        self.assertAlmostEqual(len(columns), 50, delta=8)  # 25% of 200 px, minus the soft edge
        self.assertAlmostEqual(cx, 100, delta=2)
        self.assertAlmostEqual(cy, 50, delta=2)
        self.assertEqual(int(self.bg.max()), 0)            # the original is left alone

    def test_the_cat_keeps_its_shape(self):
        out, (cx, cy) = self.tool.paste(self.bg, self.cat, 0.5, (0.5, 0.5))
        rows = np.nonzero(out[:, cx, 0] > 128)[0]
        columns = np.nonzero(out[cy, :, 0] > 128)[0]
        self.assertAlmostEqual(len(columns) / float(len(rows)), 2.0, delta=0.5)  # the crop was 40 x 20

    def test_a_cat_at_the_edge_is_moved_inside_the_picture(self):
        out, (cx, cy) = self.tool.paste(self.bg, self.cat, 0.25, (0.0, 0.0))
        self.assertGreaterEqual(cx, 24)
        self.assertGreaterEqual(cy, 12)
        out, (cx, cy) = self.tool.paste(self.bg, self.cat, 0.25, (1.0, 1.0))
        self.assertLessEqual(cx, 200 - 24)
        self.assertLessEqual(cy, 100 - 12)

    def test_window_is_centred_on_the_cat_and_has_the_size_of_the_network_input(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[48:52, 98:102] = 255                          # a marker at the centre
        crop = self.tool.window(img, (100, 50), (50, 40), 1.0)
        self.assertEqual(crop.shape, (40, 50, 3))
        self.assertEqual(int(crop[20, 25, 0]), 255)       # still in the middle
        shrunk = self.tool.window(img, (100, 50), (50, 40), 0.5)  # twice the area, shrunk to the same size
        self.assertEqual(shrunk.shape, (40, 50, 3))
        self.assertGreater(int(shrunk[20, 25, 0]), 0)

    def test_window_at_the_corner_stays_inside_the_picture(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[0, 0] = 255
        crop = self.tool.window(img, (0, 0), (50, 40), 1.0)
        self.assertEqual(crop.shape, (40, 50, 3))
        self.assertEqual(int(crop[0, 0, 0]), 255)         # the corner pixel is the crop's corner

    def test_window_never_asks_for_more_than_the_picture(self):
        img = np.zeros((30, 40, 3), np.uint8)
        crop = self.tool.window(img, (20, 15), (512, 384), 1.0)  # bigger than the picture: upscaled whole picture
        self.assertEqual(crop.shape, (384, 512, 3))

    def test_degrading_changes_the_picture_but_not_its_size(self):
        img = np.full((40, 60, 3), 100, np.uint8)
        out = self.tool.degrade(img, np.random.RandomState(1))
        self.assertEqual((out.shape, out.dtype), (img.shape, img.dtype))
        self.assertFalse((out == img).all())
        again = self.tool.degrade(img, np.random.RandomState(1))
        self.assertTrue((out == again).all())              # same seed, same picture


if __name__ == "__main__":
    unittest.main()
