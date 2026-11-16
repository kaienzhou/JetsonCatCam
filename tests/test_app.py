"""The main loop of the app with a fake backend (no camera, no web server needed)."""
import os
import tempfile
import time
import unittest

from catcam import app, web
from catcam.events import EventLog
from catcam.zones import Zone, ZoneTracker


class CountingBackend(object):
    """Hands out frames; results[i] is (frame, point, box) or None. read() takes `work` s."""

    def __init__(self, results, work=0.0):
        self.results = list(results)
        self.reads = 0
        self.work = work
        self.timing = {"detect": 12.34}

    def read(self):
        self.reads += 1
        if self.work:
            time.sleep(self.work)  # stands in for capture + detection
        return self.results[min(self.reads, len(self.results)) - 1]


class RunLoopTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.zones = [Zone("all", [[0, 0], [1, 0], [1, 1], [0, 1]])]
        self.tracker = ZoneTracker(self.zones, enter_frames=2)
        self.log = EventLog(os.path.join(self.dir, "events.jsonl"))
        self.shared = web.Shared()

    def run_loop(self, backend, frames, **kw):
        t0 = time.time()
        app.run(backend, self.tracker, self.log, self.shared, self.zones, "", set(),
                stop=lambda: backend.reads >= frames, **kw)
        return time.time() - t0

    def test_state_reports_fps_timing_and_zone(self):
        backend = CountingBackend([(None, (0.5, 0.5), (0.4, 0.4, 0.6, 0.6))])
        self.run_loop(backend, 5)
        _, state = self.shared.get()
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["zone"], "all")
        self.assertEqual(state["timing"], {"detect": 12.3})
        self.assertTrue(state["present"])

    def test_max_fps_slows_the_loop_down(self):
        backend = CountingBackend([(None, None, None)])
        elapsed = self.run_loop(backend, 6, max_fps=20)   # 6 frames at 20 fps >= 0.25 s
        self.assertGreaterEqual(elapsed, 0.24)

    def test_max_fps_counts_the_time_detection_already_took(self):
        # read() (capture + detection) takes 50 ms; the cap is 10 fps = 100 ms per frame.
        # Only the missing ~50 ms may be slept, not another full 100 ms.
        backend = CountingBackend([(None, None, None)], work=0.05)
        elapsed = self.run_loop(backend, 5, max_fps=10)
        self.assertGreaterEqual(elapsed, 0.45)
        self.assertLess(elapsed, 0.62)

    def test_no_cap_runs_flat_out(self):
        backend = CountingBackend([(None, None, None)])
        self.assertLess(self.run_loop(backend, 50), 0.2)

    def test_frames_without_a_result_do_not_crash_and_leave_status_alone(self):
        backend = CountingBackend([None])
        self.run_loop(backend, 3)
        _, state = self.shared.get()
        self.assertEqual(state["status"], "starting")  # never got a frame yet, nothing was published


if __name__ == "__main__":
    unittest.main()
