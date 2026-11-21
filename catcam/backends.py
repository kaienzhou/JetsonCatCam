"""Frame + detection sources.

read() -> (frame_bgr or None, (cx, cy) normalized cat centre or None, cat_box or None)

JetsonBackend.read() returns None instead of a tuple when no frame could be
captured (camera timeout, unplugged, end of a video file).
"""
import math
import os
import threading
import time


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _jetson_modules(need_inference=True):
    """(jetson.inference or None, jetson.utils), also under the newer jetson_* names."""
    try:
        import jetson.utils as utils
        inference = None
        if need_inference:
            import jetson.inference as inference
        return inference, utils
    except ImportError:
        pass
    try:
        import jetson_utils as utils
        inference = None
        if need_inference:
            import jetson_inference as inference
        return inference, utils
    except ImportError:
        raise RuntimeError("jetson-inference Python bindings not found "
                           "(install it first, or run with --backend fake)")


class FakeBackend(object):
    """Cat walks a loop of waypoints; for testing logic and UI without hardware.

    With OpenCV the picture is a small room (sofa, food bowl, a cat) so the web page has
    something to show. speed is in picture widths per second; try 0.3 to see events quickly
    ("fake": {"speed": 0.3} in config.json).
    """

    PATH = [(0.15, 0.2), (0.15, 0.8), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]

    def __init__(self, size=(640, 480), speed=0.05):
        self.size = size
        self.speed = speed
        self.t0 = time.time()
        self.cv2 = _cv2()
        self._room = None

    def _scene(self, cx, cy):
        import numpy as np
        cv2 = self.cv2
        w, h = self.size
        if self._room is None:
            room = np.empty((h, w, 3), np.uint8)
            room[:] = (112, 100, 92)                      # wall
            room[int(h * 0.50):] = (96, 128, 164)         # floor
            cv2.rectangle(room, (int(w * 0.05), int(h * 0.12)), (int(w * 0.45), int(h * 0.46)),
                          (84, 70, 140), -1)              # sofa
            cv2.rectangle(room, (int(w * 0.05), int(h * 0.30)), (int(w * 0.45), int(h * 0.46)),
                          (96, 84, 160), -1)              # cushion
            cv2.ellipse(room, (int(w * 0.78), int(h * 0.80)), (int(w * 0.09), int(h * 0.05)),
                        0, 0, 360, (60, 60, 190), -1)     # food bowl
            cv2.ellipse(room, (int(w * 0.78), int(h * 0.79)), (int(w * 0.06), int(h * 0.03)),
                        0, 0, 360, (90, 150, 210), -1)
            self._room = room
        frame = self._room.copy()
        x, y = int(cx * w), int(cy * h)
        fur = (40, 150, 240)
        cv2.ellipse(frame, (x, y), (int(w * 0.05), int(h * 0.04)), 0, 0, 360, fur, -1)           # body
        cv2.circle(frame, (x + int(w * 0.045), y - int(h * 0.025)), int(w * 0.027), fur, -1)     # head
        top = y - int(h * 0.045)
        for dx in (0.035, 0.057):                                                                  # ears
            ex = x + int(w * dx)
            cv2.fillPoly(frame, [np.array([(ex - 5, top), (ex + 5, top), (ex, y - int(h * 0.075))],
                                          np.int32)], fur)
        cv2.line(frame, (x - int(w * 0.05), y), (x - int(w * 0.085), y - int(h * 0.03)), fur, 4)  # tail
        return frame

    def _pos(self):
        d = (time.time() - self.t0) * self.speed
        legs = [(a, b, math.hypot(b[0] - a[0], b[1] - a[1]))
                for a, b in zip(self.PATH, self.PATH[1:] + self.PATH[:1])]
        d %= sum(l[2] for l in legs)
        for a, b, length in legs:
            if d <= length:
                f = d / length
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            d -= length

    def close(self):
        pass

    def read(self):
        time.sleep(0.1)
        cx, cy = self._pos()
        box = (cx - 0.06, cy - 0.06, cx + 0.06, cy + 0.06)
        frame = self._scene(cx, cy) if self.cv2 else None
        return frame, (cx, cy), box


class CsiCapture(object):
    """CSI camera through OpenCV + GStreamer: nvarguscamerasrc -> nvvidconv -> BGR frames.

    jetson-utils' own CSI capture breaks on some JetPack 4.6 systems when the image is
    not rotated ("nvbuf_utils: dmabuf_fd ... mapped entry NOT found", the process then
    exits). This standard pipeline does not have that problem, also lets us pick the
    sensor mode, and hands us BGR frames that can be drawn on directly.

    A thread keeps reading, so read() always returns the newest frame (never a stale
    queued one) and a stalled camera can be noticed instead of blocking the caller.
    """

    # nvvidconv flip-method for clockwise rotation in degrees
    FLIP_METHOD = {0: 0, 90: 3, 180: 2, 270: 1}

    def __init__(self, cv2, sensor_id=0, sensor_mode=3, sensor_size=(1640, 1232),
                 size=(640, 480), rate=30, rotate=0):
        self.cv2 = cv2
        out_w, out_h = size if rotate in (0, 180) else (size[1], size[0])
        self.pipeline = (
            "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
            "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1,format=NV12 ! "
            "nvvidconv flip-method=%d ! video/x-raw,width=%d,height=%d,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
            % (sensor_id, sensor_mode, sensor_size[0], sensor_size[1], rate,
               self.FLIP_METHOD[rotate], out_w, out_h))
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open the CSI camera (pipeline: %s)" % self.pipeline)
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._consumed = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout):
        """Newest frame not returned yet, or None if none arrived within `timeout` seconds.

        A frame that arrived while the caller was busy is returned at once (no waiting
        for the next one), so the caller runs as fast as its own processing allows.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._consumed, timeout):
                return None
            self._consumed = self._seq
            return self._frame

    def close(self):
        self._stop = True
        self._thread.join(2.0)
        if not self._thread.is_alive():  # releasing while read() is blocked would crash
            self.cap.release()


class JetsonBackend(object):
    """Cat detection on a camera or video file with one of two detectors.

    detector={"type": "ssd"} (default): jetson-inference detectNet, COCO ssd-mobilenet-v2.
    First start builds its TensorRT engine and takes several minutes. Finds a cat only
    when it fills roughly a quarter of the frame width or more.

    detector={"type": "yolo", "engine": "models/yolov8n_384x512.fp16.engine"}: YOLOv8
    through TensorRT (see yolo.py), much better with small / distant cats. Optional keys:
    min_conf (0.4), iou (0.5), all_classes (true: also report every other COCO class,
    for the debug overlay; false: cats only, a little faster).

    source "csi://N" (Raspberry Pi camera) is captured through OpenCV/GStreamer
    (see CsiCapture): width/height are the size of the frames used for detection and
    for the web page, sensor_mode/sensor_width/sensor_height pick the sensor mode
    (default mode 3 = 1640x1232 at 30 fps for the IMX219). If no frame arrives for
    `watchdog` seconds the process exits so a supervisor (systemd) can restart it
    together with the camera daemon.

    Any other source (USB camera "/dev/video0", a video file, ...) goes through
    jetson-utils videoSource. In jetson-inference (L4T R32.7.1) its Capture() raises
    an exception on timeout / error / end of file instead of returning None, so every
    failure is caught; after `reopen_after` failures in a row the source is reopened.
    There the frame for the web page is shrunk on the GPU to `display_width`.

    After every read(): `last_detections` lists everything the network found as
    (class name, confidence, (left, top, right, bottom) normalized), and `timing`
    holds moving averages in milliseconds for capture / detect / convert.
    """

    ROTATE_FLAG = {0: "none", 90: "clockwise", 180: "rotate-180", 270: "counterclockwise"}

    def __init__(self, source="/dev/video0", network="ssd-mobilenet-v2",
                 min_conf=0.4, width=640, height=480, rate=30, rotate=0,
                 sensor_mode=3, sensor_width=1640, sensor_height=1232,
                 display_width=640, capture_timeout=500, reopen_after=10, watchdog=20,
                 detector=None):
        if rotate not in self.ROTATE_FLAG:
            raise ValueError("rotate must be one of 0, 90, 180, 270 (clockwise degrees)")
        det = dict(detector or {})
        kind = det.pop("type", "ssd")
        if kind not in ("ssd", "yolo"):
            raise ValueError("detector type must be 'ssd' or 'yolo'")
        inference, utils = _jetson_modules(need_inference=(kind == "ssd"))
        self.utils = utils
        self.net = self.yolo = None
        if kind == "ssd":
            self.net = inference.detectNet(det.get("network", network),
                                           threshold=det.get("min_conf", min_conf))
        else:
            if _cv2() is None:
                raise RuntimeError("OpenCV (cv2) is needed for the YOLO detector")
            from .yolo import YoloTrt, CAT
            self.yolo = YoloTrt(det["engine"], utils, conf=det.get("min_conf", min_conf),
                                iou=det.get("iou", 0.5),
                                class_ids=None if det.get("all_classes") else (CAT,))
        self.source = source
        self.argv = ["--input-width=%d" % width, "--input-height=%d" % height,
                     "--input-rate=%d" % rate,
                     "--input-flip=%s" % self.ROTATE_FLAG[rotate]]
        self.display_width = display_width
        self.capture_timeout = capture_timeout
        self.reopen_after = reopen_after
        self.watchdog = watchdog
        self.cv2 = _cv2()
        self.cam = None
        self.csi = None
        self.failures = 0
        self.timing = {}
        self.last_detections = []
        self._small = None
        self._last_warn = 0.0
        self._last_frame = time.time()
        self._exit = os._exit
        if source.startswith("csi://"):
            if self.cv2 is None:
                raise RuntimeError("OpenCV (cv2) is needed for CSI cameras")
            self.csi = CsiCapture(self.cv2, sensor_id=int(source[len("csi://"):] or 0),
                                  sensor_mode=sensor_mode,
                                  sensor_size=(sensor_width, sensor_height),
                                  size=(width, height), rate=rate, rotate=rotate)
        else:
            self._open()

    def close(self):
        if self.csi is not None:
            self.csi.close()
        self.cam = None  # dropping the videoSource stops the camera pipeline cleanly

    def _warn(self, msg):
        now = time.time()
        if now - self._last_warn > 30:  # don't flood the log while the camera is missing
            self._last_warn = now
            print(msg, flush=True)

    def _tick(self, key, t0):
        ms = (time.time() - t0) * 1000.0
        old = self.timing.get(key)
        self.timing[key] = ms if old is None else 0.9 * old + 0.1 * ms

    def _open(self):
        self.cam = None  # drop the old source first so the device is released
        self.failures = 0
        try:
            self.cam = self.utils.videoSource(self.source, argv=self.argv)
        except Exception as e:
            self._warn("camera: cannot open %s: %s" % (self.source, e))

    def _frame(self, img):
        """BGR numpy frame for the web page, or None without OpenCV."""
        if not self.cv2:
            return None
        try:
            src = img
            if self.display_width and img.width > self.display_width:
                h = int(round(img.height * self.display_width / float(img.width)))
                if (self._small is None or self._small.width != self.display_width
                        or self._small.height != h):
                    self._small = self.utils.cudaAllocMapped(
                        width=self.display_width, height=h, format=img.format)
                self.utils.cudaResize(img, self._small)
                src = self._small
            self.utils.cudaDeviceSynchronize()
            rgb = self.utils.cudaToNumpy(src).copy()  # copy: the capture buffer is reused
            return self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        except Exception as e:
            self._warn("cannot read the frame as numpy: %s" % e)
            return None

    def _grab(self):
        """-> (CUDA image for the detector, BGR frame for the page or None), or None."""
        if self.csi is not None:
            t0 = time.time()
            frame = self.csi.read(self.capture_timeout / 1000.0)
            if frame is None:
                if time.time() - self._last_frame > self.watchdog:
                    print("camera: no frames for %d s, exiting so the service restarts"
                          % self.watchdog, flush=True)
                    self._exit(3)
                return None
            self._last_frame = time.time()
            self._tick("capture", t0)
            img = None  # the YOLO detector works on the BGR frame itself
            if self.yolo is None:
                t0 = time.time()
                # detectNet wants rgb8; cudaFromNumpy(isBGR=True) would only label it "bgr8"
                img = self.utils.cudaFromNumpy(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
                self._tick("convert", t0)
            return img, frame

        if self.cam is None:
            time.sleep(1.0)
            self._open()
            return None
        t0 = time.time()
        try:
            img = self.cam.Capture(timeout=self.capture_timeout)
        except Exception:  # timeout, camera unplugged, end of file
            self.failures += 1
            if self.failures >= self.reopen_after:
                self._warn("camera: no frames from %s, reopening" % self.source)
                self._open()
            return None
        self._tick("capture", t0)
        self.failures = 0
        t0 = time.time()
        frame = self._frame(img)
        self._tick("convert", t0)
        return img, frame

    def read(self):
        grabbed = self._grab()
        if grabbed is None:
            return None
        img, frame = grabbed

        t0 = time.time()
        try:
            found = self._detect(img, frame)
        except Exception as e:
            self._warn("detect error: %s" % e)
            return None
        self._tick("detect", t0)
        self.last_detections = found

        best = None
        for name, conf, box in found:
            if name == "cat" and (best is None or conf > best[0]):
                best = (conf, box)
        if best is None:
            return frame, None, None
        box = best[1]
        return frame, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box

    def _detect(self, img, frame):
        """-> [(class name, confidence, (left, top, right, bottom) normalized)]"""
        if self.yolo is not None:
            return self.yolo.detect(frame)
        w, h = img.width, img.height
        return [(self.net.GetClassDesc(d.ClassID), d.Confidence,
                 (d.Left / w, d.Top / h, d.Right / w, d.Bottom / h))
                for d in self.net.Detect(img, overlay="none")]
