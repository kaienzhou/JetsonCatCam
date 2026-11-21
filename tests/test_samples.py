"""Pictures kept by hand: the store, the web API and the main loop's part."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the other test modules, however we are started
from catcam import app, web
from catcam.config import Config
from catcam.samples import Samples
from catcam.store import Store
from catcam.zones import Zone, ZoneTracker
from test_app import CountingBackend
from test_web import WebTestCase

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

NOW = time.mktime((2026, 11, 21, 18, 15, 30, 0, 0, -1))


class SamplesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.samples = Samples(os.path.join(self.dir, "samples"), keep=3)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_sample_is_a_picture_and_what_the_detector_said(self):
        name = self.samples.save(b"\xff\xd8picture", {"zone": "sofa", "weak": [["cat", 0.21, [0, 0, 1, 1]]]}, now=NOW)
        self.assertEqual(name, "20261121-181530")
        with open(self.samples.path(name + ".jpg"), "rb") as f:
            self.assertEqual(f.read(), b"\xff\xd8picture")
        with open(self.samples.path(name + ".json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertEqual((info["zone"], info["name"], info["t"]), ("sofa", name, NOW))
        self.assertEqual(info["weak"][0][1], 0.21)
        self.assertEqual(self.samples.count(), 1)

    def test_two_in_the_same_second_both_survive(self):
        names = [self.samples.save(b"a", {}, now=NOW), self.samples.save(b"b", {}, now=NOW)]
        self.assertEqual(names, ["20261121-181530", "20261121-181530-1"])
        self.assertEqual(self.samples.count(), 2)

    def test_newest_first_and_only_the_newest_are_kept(self):
        for i in range(5):
            self.samples.save(b"x", {}, now=NOW + i)
        self.assertEqual(self.samples.latest(10), ["20261121-181534", "20261121-181533", "20261121-181532"])
        self.assertEqual(self.samples.count(), 3)
        self.assertIsNone(self.samples.path("20261121-181530.jpg"))   # the oldest are gone, json too
        self.assertIsNone(self.samples.path("20261121-181530.json"))
        self.assertEqual(sorted(os.listdir(self.samples.dir)), sorted(
            n + e for n in ("20261121-181532", "20261121-181533", "20261121-181534") for e in (".jpg", ".json")))

    def test_only_real_sample_names_are_served(self):
        name = self.samples.save(b"x", {}, now=NOW)
        self.assertIsNotNone(self.samples.path(name + ".jpg"))
        for bad in ("../x.jpg", "..", "", None, "20261121-181530.txt", "x.jpg", "20261121-181530.jpg/../../a",
                    "/etc/passwd", "20261121-181531.jpg"):  # the last one does not exist
            self.assertIsNone(self.samples.path(bad), bad)

    def test_no_half_written_files_are_left_behind(self):
        self.samples.save(b"x", {}, now=NOW)
        self.assertFalse([n for n in os.listdir(self.samples.dir) if n.endswith(".tmp")])

    def test_nothing_there_yet(self):
        empty = Samples(os.path.join(self.dir, "nope"))
        self.assertEqual((empty.count(), empty.latest()), (0, []))


class SampleWebTest(WebTestCase):
    """The web test fixture (a real server on a free port), with samples switched on."""

    def setUp(self):
        super(SampleWebTest, self).setUp()
        self.server.shutdown()          # a new server, this time with samples
        self.server.server_close()
        self.samples = Samples(os.path.join(self.dir, "samples"))
        ctx = web.Context(self.shared, self.auth, self.store, Config(self.config_path),
                          name="testcam", samples=self.samples)
        self.server = web.serve(ctx, 0, "127.0.0.1")
        self.port = self.server.server_address[1]
        self.cookie = self.login()

    def main_loop(self, picture=b"\xff\xd8frame", answer=True):
        """Plays the main loop: keeps the next picture when asked."""
        def run():
            end = time.time() + 5
            while time.time() < end:
                if self.shared.take_sample_request():
                    self.shared.set_sample_result(self.samples.save(picture, {"zone": None}) if answer else "")
                    return
                time.sleep(0.01)
        t = threading.Thread(target=run)
        t.start()
        self.addCleanup(t.join)

    def test_asking_for_a_sample_keeps_the_next_picture(self):
        self.main_loop()
        status, _, body = self.req("POST", "/api/sample", {}, cookie=self.cookie)
        self.assertEqual(status, 200, body)
        got = json.loads(body.decode("utf-8"))
        self.assertEqual((got["ok"], got["count"]), (True, 1))
        status, headers, jpeg = self.req("GET", "/sample/%s.jpg" % got["name"], cookie=self.cookie)
        self.assertEqual((status, headers["content-type"], jpeg), (200, "image/jpeg", b"\xff\xd8frame"))
        status, headers, text = self.req("GET", "/sample/%s.json" % got["name"], cookie=self.cookie)
        self.assertEqual(status, 200)
        self.assertIn("json", headers["content-type"])
        self.assertEqual(json.loads(text.decode("utf-8"))["name"], got["name"])

    def test_the_list_and_the_info_count_them(self):
        self.main_loop()
        self.req("POST", "/api/sample", {}, cookie=self.cookie)
        status, _, body = self.req("GET", "/api/samples", cookie=self.cookie)
        listing = json.loads(body.decode("utf-8"))
        self.assertEqual((status, listing["count"], len(listing["latest"])), (200, 1, 1))
        info = json.loads(self.req("GET", "/api/info", cookie=self.cookie)[2].decode("utf-8"))
        self.assertEqual(info["samples"], 1)

    def test_it_says_so_when_the_camera_has_no_picture(self):
        orig = self.shared.request_sample
        self.shared.request_sample = lambda timeout: orig(0.2)     # do not wait four seconds
        status, _, body = self.req("POST", "/api/sample", {}, cookie=self.cookie)
        self.assertEqual(status, 503)
        self.assertIn("摄像头", json.loads(body.decode("utf-8"))["error"])
        self.assertEqual(self.samples.count(), 0)
        self.assertFalse(self.shared.take_sample_request())        # nobody is left waiting for it

    def test_a_failed_save_is_reported(self):
        self.main_loop(answer=False)
        self.assertEqual(self.req("POST", "/api/sample", {}, cookie=self.cookie)[0], 503)

    def test_samples_need_a_login(self):
        self.assertEqual(self.req("POST", "/api/sample", {})[0], 401)
        self.assertEqual(self.req("GET", "/api/samples")[0], 401)
        self.assertEqual(self.req("GET", "/sample/20261121-181530.jpg")[0], 401)
        self.assertFalse(self.shared.take_sample_request())

    def test_unknown_sample_names_are_not_found(self):
        for path in ("/sample/20261121-181530.jpg", "/sample/..%2f..%2fconfig.json", "/sample/x.jpg"):
            self.assertEqual(self.req("GET", path, cookie=self.cookie)[0], 404, path)


class SampleLoopTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.dir, "catcam.db"), None)
        self.samples = Samples(os.path.join(self.dir, "samples"))
        self.shared = web.Shared()
        self.tracker = ZoneTracker([Zone("all", [[0, 0], [1, 0], [1, 1], [0, 1]])], enter_frames=2)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_loop(self, backend, frames, **kw):
        app.run(backend, self.tracker, self.store, self.shared, "", set(),
                stop=lambda: backend.reads >= frames, **kw)

    @unittest.skipUnless(cv2 is not None and np is not None, "needs OpenCV and numpy")
    def test_the_sample_is_the_full_size_picture_without_drawings_and_says_what_was_seen(self):

        class Seeing(CountingBackend):
            last_detections = [("cat", 0.55, (0.4, 0.4, 0.6, 0.6))]
            weak = [("cat", 0.2, (0.1, 0.1, 0.2, 0.2))]
            window = (0.25, 0.25, 0.75, 0.75)

            def describe(self):
                return {"engine": "m.engine"}

        def ask(n):
            if n == 2:
                threading.Thread(target=self.shared.request_sample, args=(3.0,)).start()
                time.sleep(0.05)  # the request is in before the next read() returns
        frame = np.full((96, 128, 3), 7, np.uint8)
        backend = Seeing([(frame, (0.5, 0.5), (0.4, 0.4, 0.6, 0.6))], on_read=ask)
        self.shared.last_view = time.monotonic()  # somebody watches: the frame gets drawn on
        self.run_loop(backend, 5, display_width=64, samples=self.samples)
        self.assertEqual(self.samples.count(), 1)
        name = self.samples.latest(1)[0]
        with open(self.samples.path(name + ".jpg"), "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (96, 128))            # not shrunk for the page
        self.assertLess(abs(int(img.mean()) - 7), 3)          # no zone lines, boxes or text on it
        with open(self.samples.path(name + ".json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertEqual(info["frame"], [128, 96])
        self.assertEqual(info["detections"], [["cat", 0.55, [0.4, 0.4, 0.6, 0.6]]])
        self.assertEqual(info["weak"], [["cat", 0.2, [0.1, 0.1, 0.2, 0.2]]])
        self.assertEqual(info["window"], [0.25, 0.25, 0.75, 0.75])
        self.assertEqual(info["detector"], {"engine": "m.engine"})
        self.assertEqual(info["zones"][0]["name"], "all")

    def test_without_samples_a_request_is_left_alone(self):
        self.shared._sample_request = True
        self.run_loop(CountingBackend([(None, (0.5, 0.5), (0.4, 0.4, 0.6, 0.6))]), 3)
        self.assertTrue(self.shared.take_sample_request())  # still there: nobody took it

    def test_a_frame_that_is_missing_cannot_be_kept(self):
        results = []

        def ask(n):
            if n == 2:
                threading.Thread(target=lambda: results.append(self.shared.request_sample(1.0))).start()
                time.sleep(0.05)
        self.run_loop(CountingBackend([(None, None, None)], on_read=ask), 4, samples=self.samples)
        time.sleep(0.2)
        self.assertEqual((results, self.samples.count()), ([None], 0))


if __name__ == "__main__":
    unittest.main()
