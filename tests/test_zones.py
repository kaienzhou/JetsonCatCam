import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam.zones import Zone, ZoneTracker, parse_zones, point_in_polygon, watch_window

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


def box(x0, y0, x1, y1, name="z"):
    return Zone(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class WatchWindowTest(unittest.TestCase):
    def test_a_compact_zone_gets_a_window_around_it(self):
        l, t, r, b = watch_window([box(0.5, 0.5, 0.7, 0.7)], margin=0.05)
        # 0.2 + 2 * 0.05 = 0.3 wide and high, centred on the zone
        for got, want in zip((l, t, r, b), (0.45, 0.45, 0.75, 0.75)):
            self.assertAlmostEqual(got, want)

    def test_the_window_covers_all_zones(self):
        l, t, r, b = watch_window([box(0.1, 0.1, 0.2, 0.2, "a"), box(0.5, 0.4, 0.6, 0.5, "b")], margin=0.0)
        self.assertLessEqual(l, 0.1)
        self.assertLessEqual(t, 0.1)
        self.assertGreaterEqual(r, 0.6)
        self.assertGreaterEqual(b, 0.5)

    def test_a_tiny_zone_is_not_zoomed_in_beyond_min_size(self):
        l, t, r, b = watch_window([box(0.5, 0.5, 0.52, 0.52)], margin=0.0, min_size=0.3)
        self.assertAlmostEqual(r - l, 0.3)
        self.assertAlmostEqual(b - t, 0.3)

    def test_the_window_keeps_the_aspect_and_stays_inside_the_picture(self):
        # a tall, narrow zone in the corner: the window grows sideways, then is pushed inside
        l, t, r, b = watch_window([box(0.0, 0.1, 0.1, 0.6)], aspect=1.0, margin=0.0)
        self.assertAlmostEqual((r - l), (b - t))
        self.assertGreaterEqual(l, 0.0)
        self.assertLessEqual(r, 1.0)
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(b, 1.0)

    def test_other_aspects(self):
        # network input wider than the picture: window 4/3 as wide as it is high
        l, t, r, b = watch_window([box(0.4, 0.4, 0.5, 0.5)], aspect=4 / 3.0, margin=0.05)
        self.assertAlmostEqual((r - l) / (b - t), 4 / 3.0)

    def test_zones_that_need_the_whole_picture_give_no_window(self):
        self.assertIsNone(watch_window([box(0.0, 0.0, 0.5, 1.0, "l"), box(0.5, 0.0, 1.0, 1.0, "r")]))
        self.assertIsNone(watch_window([box(0.1, 0.1, 0.9, 0.9)]))
        self.assertIsNone(watch_window([box(0.0, 0.0, 0.2, 0.2, "a"), box(0.8, 0.8, 1.0, 1.0, "b")]))

    def test_no_zones_no_window(self):
        self.assertIsNone(watch_window([]))


class SetZonesTest(unittest.TestCase):
    def test_changing_the_zones_closes_the_stay_and_the_cat_enters_the_new_one(self):
        t = make()
        for i in range(3):
            t.update(i * 0.1, (0.2, 0.2))  # settled in "sofa"
        ev = t.set_zones(5.0, [Zone("new", SQUARE)])
        self.assertEqual(types(ev), [("leave", "sofa")])
        self.assertIsNone(t.current)
        out = []
        for i in range(3):
            out += t.update(6 + i * 0.1, (0.5, 0.5))
        self.assertEqual(types(out), [("enter", "new")])  # still in view: no new "appear"

    def test_changing_the_zones_with_nobody_inside_is_quiet(self):
        t = make()
        self.assertEqual(t.set_zones(1.0, []), [])


class ParseZonesTest(unittest.TestCase):
    SQ = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]

    def test_a_good_list(self):
        zones = parse_zones([{"name": " 沙发 ", "points": self.SQ}, {"name": "食盆", "points": self.SQ[:3]}])
        self.assertEqual([z.name for z in zones], ["沙发", "食盆"])
        self.assertEqual(zones[1].to_json()["points"], [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]])

    def test_a_concave_shape_is_fine(self):
        l_shape = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.4], [0.4, 0.4], [0.4, 0.9], [0.1, 0.9]]
        self.assertEqual(len(parse_zones([{"name": "L", "points": l_shape}])), 1)

    def test_a_doubled_tap_and_a_closed_ring_are_tidied(self):
        pts = [[0.1, 0.1], [0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.1]]
        self.assertEqual(len(parse_zones([{"name": "a", "points": pts}])[0].points), 3)

    def test_every_mistake_has_a_message_that_says_what_to_fix(self):
        bow_tie = [[0.1, 0.1], [0.9, 0.6], [0.9, 0.1], [0.1, 0.4]]
        cases = [
            ("not a list", "格式"),
            ([{"name": "", "points": self.SQ}], "名字"),
            ([{"name": 5, "points": self.SQ}], "名字"),
            ([{"name": "a", "points": self.SQ}, {"name": "a", "points": self.SQ}], "重复"),
            ([{"name": "x" * 25, "points": self.SQ}], "24"),
            ([{"name": "a", "points": self.SQ[:2]}], "3 到"),
            ([{"name": "a", "points": "nope"}], "顶点"),
            ([{"name": "a", "points": [[0, 0], [1]]}], "顶点"),
            ([{"name": "a", "points": [[0, 0], [2, 0], [1, 1]]}], "超出"),
            ([{"name": "a", "points": [[0, 0], [float("nan"), 0], [1, 1]]}], "超出"),
            ([{"name": "a", "points": [[True, 0], [1, 0], [1, 1]]}], "超出"),
            ([{"name": "a", "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.1001]]}], "太小"),
            ([{"name": "a", "points": bow_tie}], "交叉"),
            ([{"name": str(i), "points": self.SQ} for i in range(13)], "最多"),
        ]
        for raw, word in cases:
            with self.assertRaises(ValueError) as cm:
                parse_zones(raw)
            self.assertIn(word, str(cm.exception), raw)


if __name__ == "__main__":
    unittest.main()
