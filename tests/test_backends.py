"""JetsonBackend logic against stub jetson / cv2 modules (no hardware needed).

The jetson stubs mimic jetson-inference L4T R32.7.1: videoSource.Capture() raises an
exception on timeout / error / end of file instead of returning None.
"""
import sys
import time
import types
import unittest


class FakeImg(object):
    width = 640
    height = 480
    format = "rgb8"


class FakeFrame(object):
    shape = (480, 640, 3)


class Det(object):
    def __init__(self, class_id, conf, left, top, right, bottom):
        self.ClassID = class_id
        self.Confidence = conf
        self.Left, self.Top, self.Right, self.Bottom = left, top, right, bottom


def install_stub(capture_script, detections, with_inference=True):
    """capture_script: list of 'ok'/'fail' consumed per Capture() call ('ok' once empty)."""
    state = {"opened": 0, "sources": [], "from_numpy": 0}

    class VideoSource(object):
        def __init__(self, uri, argv=None):
            if uri == "/dev/missing":
                raise Exception("failed to create videoSource")
            state["opened"] += 1
            state["sources"].append(uri)
            state["argv"] = argv

        def Capture(self, timeout=-1):
            step = capture_script.pop(0) if capture_script else "ok"
            if step == "fail":
                raise Exception("videoSource failed to capture image")
            return FakeImg()

    class DetectNet(object):
        def __init__(self, network, threshold=0.5):
            self.threshold = threshold

        def Detect(self, img, overlay="box,labels,conf"):
            return detections

        def GetClassDesc(self, class_id):
            return {1: "person", 17: "cat"}[class_id]

    utils = types.ModuleType("jetson.utils")
    utils.videoSource = VideoSource
    utils.cudaDeviceSynchronize = lambda: None
    def from_numpy(frame, isBGR=False):
        # detectNet only accepts rgb8/rgba8/float images: the array must already be RGB
        # (isBGR=True would just label it "bgr8" and Detect() would refuse it)
        assert not isBGR
        assert getattr(frame, "is_rgb", False), "frame was not converted to RGB first"
        state["from_numpy"] += 1
        return FakeImg()
    utils.cudaFromNumpy = from_numpy
    pkg = types.ModuleType("jetson")
    pkg.utils = utils
    sys.modules.update({"jetson": pkg, "jetson.utils": utils})
    if with_inference:
        inference = types.ModuleType("jetson.inference")
        inference.detectNet = DetectNet
        pkg.inference = inference
        sys.modules["jetson.inference"] = inference
    return state


def install_fake_cv2(frames=True, interval=0.01):
    """Fake cv2 whose VideoCapture yields a FakeFrame every `interval` s (or nothing if frames=False)."""
    cv2 = types.ModuleType("cv2")
    cv2.CAP_GSTREAMER = 1800
    cv2.COLOR_BGR2RGB = 4
    cv2.captures = []

    def cvtColor(frame, code):
        assert code == cv2.COLOR_BGR2RGB
        out = FakeFrame()
        out.is_rgb = True
        return out
    cv2.cvtColor = cvtColor

    class VideoCapture(object):
        def __init__(self, pipeline, api):
            self.pipeline = pipeline
            self.released = False
            cv2.captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            time.sleep(interval)
            return (True, FakeFrame()) if frames else (False, None)

        def release(self):
            self.released = True

    cv2.VideoCapture = VideoCapture
    sys.modules["cv2"] = cv2
    return cv2


class JetsonBackendTest(unittest.TestCase):
    def setUp(self):
        self._orig_cv2 = sys.modules.get("cv2")
        self._orig_yolo = sys.modules.get("catcam.yolo")

    def tearDown(self):
        for name in ("jetson", "jetson.inference", "jetson.utils"):
            sys.modules.pop(name, None)
        for name, orig in (("cv2", self._orig_cv2), ("catcam.yolo", self._orig_yolo)):
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    def make(self, script, detections, **kw):
        state = install_stub(script, detections)
        from catcam.backends import JetsonBackend
        b = JetsonBackend(**kw)
        b.cv2 = None  # frames are irrelevant here (and WSL has no cv2)
        return b, state

    # ---- jetson-utils path (USB camera, video files) ----

    def test_picks_most_confident_cat_and_normalizes(self):
        dets = [Det(1, 0.99, 0, 0, 100, 100),          # person: ignored
                Det(17, 0.60, 0, 0, 64, 48),
                Det(17, 0.90, 320, 240, 480, 360)]     # best cat
        b, _ = self.make([], dets)
        frame, point, box = b.read()
        self.assertIsNone(frame)
        self.assertEqual(box, (0.5, 0.5, 0.75, 0.75))
        self.assertEqual(point, (0.625, 0.625))

    def test_last_detections_list_everything_and_timing_is_recorded(self):
        dets = [Det(1, 0.70, 0, 0, 320, 240), Det(17, 0.45, 320, 240, 640, 480)]
        b, _ = self.make([], dets)
        b.read()
        self.assertEqual(b.last_detections, [("person", 0.70, (0.0, 0.0, 0.5, 0.5)),
                                             ("cat", 0.45, (0.5, 0.5, 1.0, 1.0))])
        self.assertEqual(sorted(b.timing), ["capture", "convert", "detect"])

    def test_no_cat_returns_empty_tuple_not_none(self):
        b, _ = self.make([], [Det(1, 0.9, 0, 0, 10, 10)])
        self.assertEqual(b.read(), (None, None, None))

    def test_capture_exception_returns_none(self):
        b, _ = self.make(["fail"], [])
        self.assertIsNone(b.read())
        self.assertEqual(b.failures, 1)
        self.assertEqual(b.read(), (None, None, None))  # recovered
        self.assertEqual(b.failures, 0)

    def test_reopens_after_consecutive_failures(self):
        b, state = self.make(["fail"] * 3, [], reopen_after=3)
        self.assertEqual(state["opened"], 1)
        for _ in range(3):
            self.assertIsNone(b.read())
        self.assertEqual(state["opened"], 2)
        self.assertEqual(b.failures, 0)

    def test_size_and_rotation_are_passed_to_video_source(self):
        _, state = self.make([], [], source="/dev/video0", width=1280, height=720)
        self.assertEqual(state["argv"], ["--input-width=1280", "--input-height=720",
                                         "--input-rate=30", "--input-flip=none"])

    def test_rotation_flags_for_jetson_utils_sources(self):
        flips = {}
        for rotate in (0, 90, 180, 270):
            _, state = self.make([], [], source="/dev/video0", rotate=rotate)
            flips[rotate] = state["argv"][-1]
        self.assertEqual(flips, {0: "--input-flip=none", 90: "--input-flip=clockwise",
                                 180: "--input-flip=rotate-180",
                                 270: "--input-flip=counterclockwise"})

    def test_close_releases_the_camera(self):
        b, _ = self.make([], [])
        self.assertIsNotNone(b.cam)
        b.close()
        self.assertIsNone(b.cam)

    def test_bad_rotation_is_rejected(self):
        install_stub([], [])
        from catcam.backends import JetsonBackend
        with self.assertRaises(ValueError):
            JetsonBackend(rotate=45)

    def test_missing_camera_does_not_crash_and_retries(self):
        b, state = self.make([], [], source="/dev/missing")
        self.assertIsNone(b.cam)
        orig_sleep, time.sleep = time.sleep, lambda s: None
        try:
            self.assertIsNone(b.read())  # retries opening, still missing
        finally:
            time.sleep = orig_sleep
        self.assertIsNone(b.cam)
        self.assertEqual(state["opened"], 0)

    # ---- CSI path (OpenCV + GStreamer) ----

    def make_csi(self, detections, frames=True, interval=0.01, **kw):
        cv2 = install_fake_cv2(frames, interval)
        install_stub([], detections)
        from catcam.backends import JetsonBackend
        return JetsonBackend(source="csi://0", **kw), cv2

    def test_csi_pipeline_uses_sensor_mode_and_no_flip(self):
        b, cv2 = self.make_csi([])
        try:
            p = cv2.captures[0].pipeline
            self.assertIn("nvarguscamerasrc sensor-id=0 sensor-mode=3", p)
            self.assertIn("width=1640,height=1232,framerate=30/1", p)
            self.assertIn("nvvidconv flip-method=0", p)
            self.assertIn("width=640,height=480,format=BGRx", p)
            self.assertIn("appsink drop=true max-buffers=1", p)
        finally:
            b.close()

    def test_csi_rotation_sets_flip_method_and_swaps_output_size(self):
        seen = {}
        for rotate in (90, 180, 270):
            b, cv2 = self.make_csi([], rotate=rotate)
            seen[rotate] = cv2.captures[0].pipeline
            b.close()
        self.assertIn("flip-method=3 ! video/x-raw,width=480,height=640", seen[90])
        self.assertIn("flip-method=2 ! video/x-raw,width=640,height=480", seen[180])
        self.assertIn("flip-method=1 ! video/x-raw,width=480,height=640", seen[270])

    def test_csi_read_returns_frame_and_cat(self):
        b, cv2 = self.make_csi([Det(17, 0.8, 0, 0, 320, 240)])
        try:
            frame, point, box = b.read()
            self.assertIsInstance(frame, FakeFrame)
            self.assertEqual(box, (0.0, 0.0, 0.5, 0.5))
            self.assertEqual(point, (0.25, 0.25))
            self.assertEqual(sorted(b.timing), ["capture", "convert", "detect"])
        finally:
            b.close()
        self.assertTrue(cv2.captures[0].released)

    def test_csi_returns_the_newest_frame_each_time(self):
        b, _ = self.make_csi([])
        try:
            first = b.read()[0]
            second = b.read()[0]
            self.assertIsNot(first, second)
        finally:
            b.close()

    def test_csi_frame_that_arrived_while_busy_is_returned_without_waiting(self):
        b, _ = self.make_csi([], interval=0.2)
        try:
            b.csi.read(2.0)             # first frame
            time.sleep(0.5)             # "processing": a newer frame arrives meanwhile
            t0 = time.time()
            self.assertIsNotNone(b.csi.read(2.0))
            self.assertLess(time.time() - t0, 0.1)  # not a whole frame interval of waiting
        finally:
            b.close()

    # ---- YOLO detector (TensorRT part faked) ----

    def make_yolo(self, found, detector=None, **kw):
        install_fake_cv2(True, 0.01)
        state = install_stub([], [], with_inference=False)  # YOLO must not need jetson.inference
        created = {}
        fake = types.ModuleType("catcam.yolo")
        fake.CAT = 15

        class YoloTrt(object):
            def __init__(self, engine, utils, conf, iou, class_ids):
                created.update(engine=engine, conf=conf, iou=iou, class_ids=class_ids)

            def detect(self, frame):
                return found
        fake.YoloTrt = YoloTrt
        sys.modules["catcam.yolo"] = fake
        from catcam.backends import JetsonBackend
        det = {"type": "yolo", "engine": "m.engine"}
        det.update(detector or {})
        return JetsonBackend(source="csi://0", detector=det, **kw), created, state

    def test_yolo_gets_engine_and_options(self):
        b, created, _ = self.make_yolo([])
        b.close()
        self.assertEqual(created, {"engine": "m.engine", "conf": 0.4, "iou": 0.5, "class_ids": (15,)})
        b, created, _ = self.make_yolo([], detector={"min_conf": 0.3, "iou": 0.6, "all_classes": True})
        b.close()
        self.assertEqual(created, {"engine": "m.engine", "conf": 0.3, "iou": 0.6, "class_ids": None})

    def test_yolo_read_picks_the_best_cat(self):
        found = [("dog", 0.9, (0.0, 0.0, 0.2, 0.2)),
                 ("cat", 0.6, (0.1, 0.2, 0.3, 0.4)),
                 ("cat", 0.8, (0.5, 0.5, 0.7, 0.9))]
        b, _, state = self.make_yolo(found)
        try:
            frame, point, box = b.read()
        finally:
            b.close()
        self.assertIsInstance(frame, FakeFrame)
        for got, want in zip(box, (0.5, 0.5, 0.7, 0.9)):
            self.assertAlmostEqual(got, want)
        for got, want in zip(point, (0.6, 0.7)):
            self.assertAlmostEqual(got, want)
        self.assertEqual([n for n, _, _ in b.last_detections], ["dog", "cat", "cat"])
        self.assertEqual(state["from_numpy"], 0)  # no CUDA image is built for YOLO

    def test_yolo_without_cat_returns_the_frame_only(self):
        b, _, _ = self.make_yolo([("dog", 0.9, (0.0, 0.0, 0.2, 0.2))])
        try:
            frame, point, box = b.read()
        finally:
            b.close()
        self.assertIsInstance(frame, FakeFrame)
        self.assertEqual((point, box), (None, None))

    def test_unknown_detector_type_is_rejected(self):
        install_stub([], [])
        from catcam.backends import JetsonBackend
        with self.assertRaises(ValueError):
            JetsonBackend(detector={"type": "nope"})

    def test_csi_watchdog_exits_when_frames_stop(self):
        b, _ = self.make_csi([], frames=False, watchdog=0.1, capture_timeout=20)
        codes = []

        def fake_exit(code):
            codes.append(code)
            raise SystemExit(code)
        b._exit = fake_exit
        try:
            with self.assertRaises(SystemExit):
                for _ in range(100):
                    self.assertIsNone(b.read())
        finally:
            b.close()
        self.assertEqual(codes, [3])


if __name__ == "__main__":
    unittest.main()
