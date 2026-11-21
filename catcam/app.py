import argparse
import os
import signal
import socket
import time

from . import __version__, notify, web
from .auth import Auth
from .backends import FakeBackend, JetsonBackend
from .config import Config
from .store import SNAP_TYPES, Store
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


class StateCache(object):
    """Today's totals and the latest events come from SQLite.

    They are read again when an event happened, when the day changes, or every 30 s, not on
    every frame.
    """

    def __init__(self, store):
        self.store = store
        self.totals = {}
        self.visits = {}
        self.recent = []
        self.day = None
        self.day_start = 0.0
        self.loaded = 0.0
        self.dirty = True

    def refresh(self, now):
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if not (self.dirty or day != self.day or now - self.loaded > 30):
            return
        try:
            today = self.store.history(1, now)[0]
            self.recent = self.store.recent(40)
        except Exception as exc:  # keep showing the old numbers
            print("could not read the history:", exc, flush=True)
            self.loaded = now
            return
        self.totals = {z: v["seconds"] for z, v in today["zones"].items()}
        self.visits = {z: v["visits"] for z, v in today["zones"].items()}
        self.day, self.day_start, self.loaded, self.dirty = day, today["t0"], now, False


def make_state(tracker, cache, status, now, fps=0.0, timing=None):
    """status: 'starting' (loading the model), 'ok', or 'no_camera' (no frames for 5 s)."""
    cache.refresh(now)
    totals, visits = dict(cache.totals), dict(cache.visits)
    if tracker.current is not None and tracker.entered_at is not None:
        # a stay is written to the database when the cat leaves, so add the one in progress
        start = max(tracker.entered_at, cache.day_start)
        totals[tracker.current] = totals.get(tracker.current, 0.0) + max(0.0, now - start)
        if tracker.entered_at >= cache.day_start:
            visits[tracker.current] = visits.get(tracker.current, 0) + 1
    state = {"now": now, "status": status, "fps": round(fps, 1), "present": tracker.present,
             "zone": tracker.current, "since": tracker.entered_at,
             "totals": {k: round(v, 1) for k, v in totals.items()}, "visits": visits,
             "events": cache.recent, "zones": [z.name for z in tracker.zones]}
    if timing:
        state["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return state


def _record(store, cache, events, jpeg, notify_url, notify_zones):
    for ev in events:
        try:
            store.add(ev, jpeg if ev["type"] in SNAP_TYPES else None)
        except Exception as exc:  # a full disk or a bad card must not stop the camera
            print("could not save an event:", exc, flush=True)
        print(time.strftime("%H:%M:%S"), LABEL[ev["type"]], ev["zone"] or "", flush=True)
        if ev["type"] == "enter" and ev["zone"] in notify_zones:
            notify.send(notify_url, "猫咪", "猫进入了 " + ev["zone"])
    if events:
        cache.dirty = True


def run(backend, tracker, store, shared, notify_url, notify_zones, debug=False, max_fps=0,
        stop=lambda: False, keep_days=30, max_snap_bytes=500 * 1024 * 1024):
    """Main loop. max_fps > 0 limits the loop rate (less heat and power draw);
    stop() is polled once per iteration (tests use it to end the loop).

    Pictures are only drawn and encoded while somebody watches (or when an event needs a
    snapshot): an unattended camera costs less power.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None

    status = "ok"
    last_frame = time.time()
    fps = 0.0
    min_dt = 1.0 / max_fps if max_fps else 0.0
    cache = StateCache(store)
    next_purge = 0.0
    while not stop():
        t_iter = time.time()

        request = shared.take_zone_request()  # the zone editor saved something
        if request is not None:
            _record(store, cache, tracker.set_zones(t_iter, request), None, notify_url, notify_zones)
        if time.monotonic() >= next_purge:
            next_purge = time.monotonic() + 3600
            try:
                removed = store.purge(keep_days, max_snap_bytes)
                if removed:
                    print("removed %d old pictures" % removed, flush=True)
            except Exception as exc:
                print("clean-up failed:", exc, flush=True)

        result = backend.read()
        now = time.time()
        if result is None:  # no frame: camera timeout or reconnecting
            if status == "ok" and now - last_frame > 5:
                status = "no_camera"
                fps = 0.0
                shared.set_state(make_state(tracker, cache, status, now))
            continue
        dt = now - last_frame
        if 0 < dt < 2:  # ignore gaps (startup, camera glitches)
            fps = 1.0 / dt if not fps else 0.9 * fps + 0.1 / dt
        last_frame = now
        status = "ok"
        frame, point, box = result

        events = tracker.update(now, point)

        jpeg = None
        watching = shared.wants_frame()
        need_snapshot = any(e["type"] in SNAP_TYPES for e in events)
        if cv2 is not None and frame is not None and (watching or need_snapshot or shared.wants_raw()):
            if shared.wants_raw():  # the zone editor wants the picture without our drawings
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    shared.set_raw(buf.tobytes())
            dets = getattr(backend, "last_detections", None) if debug else None
            draw(cv2, frame, tracker.zones, box, tracker.current, dets)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                jpeg = buf.tobytes()
                if watching:
                    shared.set_frame(jpeg)

        _record(store, cache, events, jpeg, notify_url, notify_zones)
        shared.set_state(make_state(tracker, cache, status, now, fps, getattr(backend, "timing", None)))

        spent = time.time() - t_iter  # the whole iteration, detection included
        if spent < min_dt:  # frame cap: the next read() then returns the newest frame at once
            time.sleep(min_dt - spent)


def _lan_ip():
    """The address this machine uses on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def show_code(auth, port):
    if not auth.enabled:
        print("口令登录已在 config.json 里关闭（auth.enabled 为 false）")
        return
    ip = _lan_ip() or "<设备的 IP 地址>"
    print("访问口令：%s" % auth.code)
    print("在手机或电脑的浏览器里打开：http://%s:%d/login#code=%s" % (ip, port, auth.code))
    print("（链接里带着口令，只会自动登录，不会发给任何网站；别把它发给别人）")


def main():
    ap = argparse.ArgumentParser(prog="catcam")
    ap.add_argument("command", nargs="?", choices=["run", "code", "new-code"], default="run",
                    help="run (default); code = show the access code; new-code = make a new one")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--backend", choices=["fake", "jetson"], default="jetson")
    args = ap.parse_args()

    config = Config(args.config)
    cfg = config.load()
    port = cfg.get("port", 8080)
    data_dir = cfg.get("data_dir", "data")
    auth_cfg = cfg.get("auth", {})
    auth = Auth(data_dir, code=auth_cfg.get("code", ""), enabled=auth_cfg.get("enabled", True))

    if args.command == "code":
        show_code(auth, port)
        return
    if args.command == "new-code":
        try:
            auth.rotate()
        except ValueError as exc:
            raise SystemExit(str(exc))
        print("已换新口令，所有已登录的手机和电脑都要重新登录。")
        show_code(auth, port)
        return

    zones = [Zone(z["name"], z["points"]) for z in cfg["zones"]]
    tracker = ZoneTracker(zones, **cfg.get("tracker", {}))
    snap_cfg = cfg.get("snapshots", {})
    snap_dir = os.path.join(data_dir, "snapshots") if snap_cfg.get("enabled", True) else None
    store = Store(os.path.join(data_dir, "catcam.db"), snap_dir)
    imported = store.import_jsonl(cfg.get("log", os.path.join(data_dir, "events.jsonl")))
    if imported:
        print("imported %d events from the old events.jsonl" % imported, flush=True)
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
    shared.zones_json = [z.to_json() for z in zones]
    shared.set_state(make_state(tracker, StateCache(store), "starting", time.time()))
    web.serve(web.Context(shared, auth, store, config), port)
    print("catcam %s: web ui on port %d" % (__version__, port), flush=True)
    if auth.created:
        print("=" * 60)
        print("第一次启动，已生成访问口令：%s" % auth.code)
        print("随时查看：python3 -m catcam code   （在项目目录里执行）")
        print("=" * 60, flush=True)

    if args.backend == "fake":
        backend = FakeBackend(**cfg.get("fake", {}))
    else:
        backend = JetsonBackend(detector=cfg.get("detector"), **cfg.get("camera", {}))

    try:
        run(backend, tracker, store, shared, notify_url, notify_zones,
            debug=cfg.get("debug", False), max_fps=cfg.get("max_fps", 0),
            keep_days=snap_cfg.get("keep_days", 30),
            max_snap_bytes=int(snap_cfg.get("max_mb", 500)) * 1024 * 1024)
    finally:
        backend.close()
        store.close()


if __name__ == "__main__":
    main()
