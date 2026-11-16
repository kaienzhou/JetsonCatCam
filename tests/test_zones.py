import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam.zones import Zone, ZoneTracker, point_in_polygon

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def make():
    zones = [Zone("sofa", [(0, 0), (0.4, 0), (0.4, 0.4), (0, 0.4)]),
             Zone("bowl", [(0.6, 0.6), (1, 0.6), (1, 1), (0.6, 1)])]
    return ZoneTracker(zones, enter_frames=3, leave_seconds=2.0, lost_seconds=4.0)


def types(events):
    return [(e["type"], e["zone"]) for e in events]


class PolygonTest(unittest.TestCase):
    def test_inside_outside(self):
        self.assertTrue(point_in_polygon(0.5, 0.5, SQUARE))
        self.assertFalse(point_in_polygon(1.5, 0.5, SQUARE))

    def test_concave(self):
        l_shape = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
        self.assertTrue(point_in_polygon(0.2, 0.8, l_shape))
        self.assertFalse(point_in_polygon(0.8, 0.8, l_shape))


class TrackerTest(unittest.TestCase):
    def test_enter_needs_consecutive_frames(self):
        t = make()
        self.assertEqual(types(t.update(0, (0.2, 0.2))), [("appear", None)])
        self.assertEqual(t.update(0.1, (0.2, 0.2)), [])
        self.assertEqual(types(t.update(0.2, (0.2, 0.2))), [("enter", "sofa")])

    def test_flicker_does_not_enter(self):
        t = make()
        t.update(0, (0.2, 0.2))
        t.update(0.1, (0.2, 0.2))
        t.update(0.2, (0.5, 0.5))  # one frame outside resets the count
        self.assertEqual(t.update(0.3, (0.2, 0.2)), [])
        self.assertIsNone(t.current)

    def test_leave_to_open_floor_after_grace(self):
        t = make()
        for i in range(3):
            t.update(i * 0.1, (0.2, 0.2))
        self.assertEqual(t.update(1.0, (0.5, 0.5)), [])  # within grace
        ev = t.update(2.5, (0.5, 0.5))
        self.assertEqual(types(ev), [("leave", "sofa")])
        self.assertEqual(ev[0]["duration"], 2.3)

    def test_direct_switch(self):
        t = make()
        for i in range(3):
            t.update(i * 0.1, (0.2, 0.2))
        out = []
        for i in range(3):
            out += t.update(5 + i * 0.1, (0.8, 0.8))
        self.assertEqual(types(out), [("leave", "sofa"), ("enter", "bowl")])

    def test_lost_closes_stay(self):
        t = make()
        for i in range(3):
            t.update(i * 0.1, (0.2, 0.2))
        self.assertEqual(t.update(1, None), [])
        ev = t.update(5, None)
        self.assertEqual(types(ev), [("leave", "sofa"), ("lost", None)])
        self.assertFalse(t.present)

    def test_reappear(self):
        t = make()
        t.update(0, (0.5, 0.5))
        t.update(10, None)
        self.assertEqual(types(t.update(20, (0.5, 0.5))), [("appear", None)])


if __name__ == "__main__":
    unittest.main()
