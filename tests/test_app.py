"""The main loop with a fake backend (no camera, no web server needed)."""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam import app, web
from catcam.store import Store
from catcam.zones import Zone, ZoneTracker

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
INSIDE = (0.5, 0.5)
BOX = (0.4, 0.4, 0.6, 0.6)


class CountingBackend(object):
    """Hands out frames; results[i] is (frame, point, box) or None. read() takes `work` s."""

    def __init__(self, results, work=0.0, on_read=None):
        self.results = list(results)
        self.reads = 0
        self.work = work
        self.on_read = on_read
        self.timing = {"detect": 12.34}

    def read(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        if self.work:
            time.sleep(self.work)  # stands in for capture + detection
        return self.results[min(self.reads, len(self.results)) - 1]


class RunLoopTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.tracker = ZoneTracker([Zone("all", SQUARE)], enter_frames=2)
        self.store = Store(os.path.join(self.dir, "catcam.db"), os.path.join(self.dir, "snaps"))
        self.shared = web.Shared()

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_loop(self, backend, frames, **kw):
        t0 = time.time()
        app.run(backend, self.tracker, self.store, self.shared, "", set(),
                stop=lambda: backend.reads >= frames, **kw)
        return time.time() - t0


class RunLoopTest(RunLoopTestCase):
    def test_state_reports_fps_timing_zone_and_the_stay_in_progress(self):
        self.run_loop(CountingBackend([(None, INSIDE, BOX)]), 5)
        state = self.shared.get_state()
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["zone"], "all")
        self.assertEqual(state["zones"], ["all"])
        self.assertEqual(state["timing"], {"detect": 12.3})
        self.assertTrue(state["present"])
        self.assertIsNotNone(state["since"])
        self.assertEqual(state["visits"], {"all": 1})  # the visit in progress counts
        self.assertIn("all", state["totals"])

    def test_events_go_into_the_history_and_into_the_state(self):
        self.run_loop(CountingBackend([(None, INSIDE, BOX)]), 5)
        self.assertEqual([e["type"] for e in self.store.recent()], ["appear", "enter"])
        self.assertEqual([e["type"] for e in self.shared.get_state()["events"]], ["appear", "enter"])

    def test_max_fps_slows_the_loop_down(self):
        elapsed = self.run_loop(CountingBackend([(None, None, None)]), 6, max_fps=20)  # 6 frames at 20 fps
        self.assertGreaterEqual(elapsed, 0.24)

    def test_max_fps_counts_the_time_detection_already_took(self):
        # read() (capture + detection) takes 50 ms; the cap is 10 fps = 100 ms per frame.
        # Only the missing ~50 ms may be slept, not another full 100 ms.
        elapsed = self.run_loop(CountingBackend([(None, None, None)], work=0.05), 5, max_fps=10)
        self.assertGreaterEqual(elapsed, 0.45)
        self.assertLess(elapsed, 0.62)

    def test_no_cap_runs_flat_out(self):
        self.assertLess(self.run_loop(CountingBackend([(None, None, None)]), 50), 0.2)

    def test_frames_without_a_result_do_not_crash_and_leave_status_alone(self):
        self.run_loop(CountingBackend([None]), 3)
        self.assertEqual(self.shared.get_state()["status"], "starting")  # never got a frame, nothing published

    def test_zones_saved_in_the_editor_take_effect_between_frames(self):
        def save_new_zones(n):
            if n == 4:
                self.shared.request_zones([Zone("new", SQUARE)])
        self.run_loop(CountingBackend([(None, INSIDE, BOX)], on_read=save_new_zones), 8)
        pairs = [(e["type"], e["zone"]) for e in self.store.recent()]
        self.assertEqual(pairs, [("appear", None), ("enter", "all"), ("leave", "all"), ("enter", "new")])
        self.assertEqual(self.shared.get_state()["zones"], ["new"])

    def test_a_failing_database_does_not_stop_the_camera(self):
        self.store.close()  # every write now fails
        self.run_loop(CountingBackend([(None, INSIDE, BOX)]), 4)
        self.assertEqual(self.shared.get_state()["status"], "ok")


@unittest.skipUnless(cv2 is not None and np is not None, "needs OpenCV and numpy")
class PictureTest(RunLoopTestCase):
    def frames(self):
        return CountingBackend([(np.zeros((48, 64, 3), np.uint8), INSIDE, BOX)])

    def test_nobody_watching_means_no_stream_pictures_but_events_keep_a_snapshot(self):
        self.run_loop(self.frames(), 4)
        self.assertIsNone(self.shared.image)
        enter = [e for e in self.store.recent() if e["type"] == "enter"][0]
        path = self.store.snapshot_path(enter["id"])
        self.assertTrue(path and os.path.getsize(path) > 100)

    def test_somebody_watching_gets_pictures(self):
        self.shared.last_view = time.monotonic()
        self.run_loop(self.frames(), 4)
        self.assertTrue(self.shared.image.startswith(b"\xff\xd8"))  # a JPEG

    def test_the_editor_gets_a_picture_without_drawings(self):
        self.shared.raw_until = time.monotonic() + 30
        self.run_loop(self.frames(), 4)
        self.assertTrue(self.shared.raw.startswith(b"\xff\xd8"))


class MakeStateTest(unittest.TestCase):
    def test_a_stay_that_began_yesterday_counts_only_from_midnight(self):
        d = tempfile.mkdtemp()
        store = Store(os.path.join(d, "catcam.db"), None)
        try:
            now = time.mktime((2026, 11, 16, 1, 0, 0, 0, 0, -1))  # 01:00
            tracker = ZoneTracker([Zone("all", SQUARE)])
            tracker.current, tracker.entered_at, tracker.present = "all", now - 3 * 3600, True  # since 22:00
            state = app.make_state(tracker, app.StateCache(store), "ok", now)
            self.assertEqual(state["totals"], {"all": 3600.0})  # 00:00 to 01:00
            self.assertEqual(state["visits"], {})  # the visit began yesterday
        finally:
            store.close()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
