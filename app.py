import argparse
import json
import signal
import time

from . import notify, web
from .backends import FakeBackend, JetsonBackend
from .events import EventLog
from .zones import Zone, ZoneTracker

LABEL = {"appear": "出现", "lost": "离开视野", "enter": "进入", "leave": "离开"}


def draw(cv2, frame, zones, box, current, detections=None):
    import numpy as np
    h, w = frame.shape[:2]
    for z in zones:
        pts = [(int(x * w), int(y * h)) for x, y in z.points]
        color = (0, 220, 0) if z.name == current else (200, 200, 200)
        cv2.polylines(frame, [np.array(pts, "int32")], True, color, 2)
        cv2.putText(frame, z.name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for name, conf, (l, t, r, b) in detections or []:  # debug: everything the network found
        p1 = (int(l * w), int(t * h))
        cv2.rectangle(frame, p1, (int(r * w), int(b * h)), (0, 255, 255), 1)
        cv2.putText(frame, "%s %.2f" % (name, conf), (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if box:
        cv2.rectangle(frame, (int(box[0] * w), int(box[1] * h)),
                      (int(box[2] * w), int(box[3] * h)), (0, 160, 255), 2)


def make_state(tracker, log, status, fps=0.0, timing=None):
    """status: 'starting' (loading the model), 'ok', or 'no_camera' (no frames for 5 s)."""
    state = {"status": status, "fps": round(fps, 1), "present": tracker.present,
             "zone": tracker.current, "totals": log.today_totals(),
             "events": list(log.recent)[-40:]}
    if timing:
        state["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return state


def run(backend, tracker, log, shared, zones, notify_url, notify_zones, debug=False,
        max_fps=0, stop=lambda: False):
    """Main loop. max_fps > 0 limits the loop rate (less heat and power draw);
    stop() is polled once per iteration (tests use it to end the loop)."""
    try:
        import cv2
    except ImportError:
        cv2 = None

    status = "ok"
    last_frame = time.time()
    fps = 0.0
    min_dt = 1.0 / max_fps if max_fps else 0.0
    while not stop():
        t_iter = time.time()
        result = backend.read()
        now = time.time()
        if result is None:  # no frame: camera timeout or reconnecting
            if status == "ok" and now - last_frame > 5:
                status = "no_camera"
                fps = 0.0
                shared.set_state(make_state(tracker, log, status))
            continue
        dt = now - last_frame
        if 0 < dt < 2:  # ignore gaps (startup, camera glitches)
            fps = 1.0 / dt if not fps else 0.9 * fps + 0.1 / dt
        last_frame = now
        status = "ok"
        frame, point, box = result

        for ev in tracker.update(now, point):
            log.add(ev)
            print(time.strftime("%H:%M:%S"), LABEL[ev["type"]], ev["zone"] or "", flush=True)
            if ev["type"] == "enter" and ev["zone"] in notify_zones:
                notify.send(notify_url, "猫咪", "猫进入了 " + ev["zone"])

        shared.set_state(make_state(tracker, log, status, fps, getattr(backend, "timing", None)))

        if cv2 is not None and frame is not None:
            dets = getattr(backend, "last_detections", None) if debug else None
            draw(cv2, frame, zones, box, tracker.current, dets)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                shared.set_frame(buf.tobytes())

        spent = time.time() - t_iter  # the whole iteration, detection included
        if spent < min_dt:  # frame cap: the next read() then returns the newest frame at once
            time.sleep(min_dt - spent)


def main():
    ap = argparse.ArgumentParser(prog="catcam")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--backend", choices=["fake", "jetson"], default="jetson")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    zones = [Zone(z["name"], z["points"]) for z in cfg["zones"]]
    tracker = ZoneTracker(zones, **cfg.get("tracker", {}))
    log = EventLog(cfg.get("log", "data/events.jsonl"))
    notify_zones = set(cfg.get("notify", {}).get("zones", []))
    notify_url = cfg.get("notify", {}).get("url", "")

    # SIGTERM (systemctl stop, kill) must unwind normally so the camera is closed
    # cleanly; a killed Argus session can leave nvargus-daemon unable to open the
    # camera again ("Failed to create CaptureSession").
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)

    # Start the web page first: the first model load builds a TensorRT engine
    # (minutes) and the page shows "starting" meanwhile instead of refusing connections.
    shared = web.Shared()
    web.serve(shared, cfg.get("port", 8080))
    print("web ui on port", cfg.get("port", 8080), flush=True)

    if args.backend == "fake":
        backend = FakeBackend()
    else:
        backend = JetsonBackend(detector=cfg.get("detector"), **cfg.get("camera", {}))

    try:
        run(backend, tracker, log, shared, zones, notify_url, notify_zones,
            debug=cfg.get("debug", False), max_fps=cfg.get("max_fps", 0))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
