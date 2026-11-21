"""Standard-library web server: login, JSON API, MJPEG stream and the web app (catcam/static).

No Flask: JetPack 4.6 ships Python 3.6 and we want zero pip installs.

Everything except the login page, the app's own files and /api/hello needs the access code
(see auth.py). The camera only encodes pictures while somebody is looking (Shared.wants_frame),
which keeps the board cooler when nobody is watching.
"""
import csv
import datetime
import io
import json
import os
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from . import __version__
from .zones import parse_zones

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# URL path -> (file, content type, login needed). The app's own files are public: they are open
# source and the browser fetches the manifest without cookies.
PAGES = {
    "/": ("index.html", "text/html; charset=utf-8", True),
    "/login": ("login.html", "text/html; charset=utf-8", False),
    "/app.js": ("app.js", "application/javascript; charset=utf-8", False),
    "/app.css": ("app.css", "text/css; charset=utf-8", False),
    "/login.js": ("login.js", "application/javascript; charset=utf-8", False),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json", False),
    "/icon-192.png": ("icon-192.png", "image/png", False),
    "/icon-512.png": ("icon-512.png", "image/png", False),
    "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png", False),
    "/favicon.ico": ("icon-192.png", "image/png", False),
}

CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
       "form-action 'self'")

MAX_VIEWERS = 6        # simultaneous video streams
SNAP_URL = re.compile(r"^/snap/(\d{1,12})\.jpg$")
SAMPLE_URL = re.compile(r"^/sample/([0-9-]{15,19}\.(?:jpg|json))$")


class Shared(object):
    """What the main loop publishes and the HTTP threads read."""

    def __init__(self):
        self.cond = threading.Condition()
        self.image = None          # latest annotated JPEG (bytes)
        self.seq = 0
        self.image_time = 0.0
        self.raw = None            # latest JPEG without drawings, for the zone editor
        self.raw_time = 0.0
        self.raw_watch = None      # the part of that picture the detector looks at, or None
        self.state = {"status": "starting", "present": False, "zone": None, "since": None,
                      "totals": {}, "visits": {}, "events": [], "zones": []}
        self.viewers = 0
        self.last_view = -1e9      # monotonic time of the last picture request (never, so far)
        self.debug = False         # draw what the detector sees on the pictures
        self.raw_until = 0.0
        self.zones_json = []       # the zones in force, as the editor shows them
        self._zone_request = None
        self._sample_request = False
        self._sample_result = None

    # -- written by the main loop

    def set_frame(self, jpeg):
        with self.cond:
            self.image = jpeg
            self.image_time = time.monotonic()
            self.seq += 1
            self.cond.notify_all()

    def set_raw(self, jpeg, watch=None):
        with self.cond:
            self.raw = jpeg
            self.raw_watch = watch
            self.raw_time = time.monotonic()
            self.cond.notify_all()

    def set_state(self, state):
        with self.cond:
            self.state = state

    def wants_frame(self):
        """Is somebody watching? Then the main loop draws and encodes pictures."""
        return self.viewers > 0 or time.monotonic() - self.last_view < 5.0

    def wants_raw(self):
        return time.monotonic() < self.raw_until

    def take_zone_request(self):
        with self.cond:
            req, self._zone_request = self._zone_request, None
            return req

    def take_sample_request(self):
        """Did somebody ask to keep the next picture? (asking again is answered with the same one)"""
        with self.cond:
            req, self._sample_request = self._sample_request, False
            return req

    def set_sample_result(self, name):
        with self.cond:
            self._sample_result = name or ""
            self.cond.notify_all()

    # -- used by the HTTP threads

    def get_state(self):
        with self.cond:
            return self.state

    def add_viewer(self):
        with self.cond:
            if self.viewers >= MAX_VIEWERS:
                return False
            self.viewers += 1
            return True

    def remove_viewer(self):
        with self.cond:
            self.viewers = max(0, self.viewers - 1)

    def request_zones(self, zones):
        with self.cond:
            self._zone_request = zones

    def request_sample(self, timeout):
        """Ask the main loop to keep the next camera picture. Returns its name, None if that failed."""
        end = time.monotonic() + timeout
        with self.cond:
            self._sample_result = None
            self._sample_request = True
            while self._sample_result is None:
                left = end - time.monotonic()
                if left <= 0:
                    self._sample_request = False
                    return None
                self.cond.wait(left)
            return self._sample_result or None

    def wait_frame(self, last_seq, timeout):
        """Block until a picture newer than last_seq exists. Returns (seq, jpeg) or (last_seq, None)."""
        end = time.monotonic() + timeout
        with self.cond:
            while self.image is None or self.seq == last_seq:
                left = end - time.monotonic()
                if left <= 0:
                    return last_seq, None
                self.cond.wait(left)
            return self.seq, self.image

    def get_image(self, max_age, timeout):
        """A recent annotated picture; asks the main loop for one if needed."""
        self.last_view = time.monotonic()
        return self._fresh("image", "image_time", max_age, timeout)

    def get_raw(self, max_age, timeout):
        """A recent picture without drawings; asks the main loop for one if needed."""
        self.raw_until = time.monotonic() + 6.0
        return self._fresh("raw", "raw_time", max_age, timeout)

    def _fresh(self, attr, stamp, max_age, timeout):
        end = time.monotonic() + timeout
        with self.cond:
            while True:
                value = getattr(self, attr)
                if value is not None and time.monotonic() - getattr(self, stamp) <= max_age:
                    return value
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(left)


class Context(object):
    """Everything the request handlers need."""

    def __init__(self, shared, auth, store, config, name=None, samples=None):
        self.shared = shared
        self.auth = auth
        self.store = store
        self.config = config
        self.samples = samples
        self.name = name or socket.gethostname()
        self.started = time.time()


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, socket.timeout)):
            return  # a phone went to sleep mid-stream: normal
        HTTPServer.handle_error(self, request, client_address)


def _int(query, name, default, lo, hi):
    try:
        value = int(query.get(name, [default])[0])
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, value))


def _open_stay(state):
    if state.get("zone") and state.get("since") is not None:
        return (state["zone"], state["since"])
    return None


def _make_handler(ctx):
    shared, auth, store = ctx.shared, ctx.auth, ctx.store

    class Handler(BaseHTTPRequestHandler):
        server_version = "catcam/" + __version__
        sys_version = ""
        timeout = 30  # a stalled client must not hold a thread forever

        def log_message(self, *args):
            pass

        # ---- helpers

        def _ip(self):
            return self.client_address[0]

        def _send(self, code, ctype, body, extra=None, cache="no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for key, value in extra or []:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj, extra=None):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send(code, "application/json; charset=utf-8", body, extra)

        def _redirect(self, url):
            self.send_response(302)
            self.send_header("Location", url)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _cookie(self, name):
            for part in self.headers.get("Cookie", "").split(";"):
                key, _, value = part.strip().partition("=")
                if key == name:
                    return value
            return ""

        def _authed(self):
            if not auth.enabled:
                return True
            bearer = self.headers.get("Authorization", "")
            if bearer.startswith("Bearer ") and auth.check_code(bearer[7:].strip()):
                return True
            return auth.check_cookie(self._cookie(auth.COOKIE))

        def _same_origin(self):
            """A page on another site must not be able to make the browser post here."""
            origin = self.headers.get("Origin")
            return not origin or urlparse(origin).netloc == self.headers.get("Host", "")

        def _body(self):
            """The JSON object of a POST, or None after answering with an error."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": "send JSON"})
                return None
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > 65536:
                self._json(413, {"error": "body too large"})
                return None
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "not valid JSON"})
                return None
            if not isinstance(data, dict):
                self._json(400, {"error": "expected a JSON object"})
                return None
            return data

        # ---- GET

        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if path in PAGES:
                return self._page(path)
            if path == "/api/hello":  # lets a client recognise a camera on the network
                return self._json(200, {"app": "catcam", "api": 1, "version": __version__,
                                        "name": ctx.name, "login": auth.enabled})
            if not self._authed():
                return self._json(401, {"error": "login required"})
            routes = {
                "/api/state": lambda: self._json(200, shared.get_state()),
                "/api/info": self._info,
                "/api/history": lambda: self._history(query),
                "/api/day": lambda: self._day(query),
                "/api/zones": lambda: self._json(200, {"zones": shared.zones_json}),
                "/api/frame.jpg": lambda: self._frame(query),
                "/api/export.csv": lambda: self._export(query),
                "/api/samples": self._samples,
                "/stream.mjpg": self._stream,
            }
            handler = routes.get(path)
            if handler:
                return handler()
            m = SNAP_URL.match(path)
            if m:
                return self._snapshot(int(m.group(1)))
            m = SAMPLE_URL.match(path)
            if m:
                return self._sample_file(m.group(1))
            self._json(404, {"error": "not found"})

        def _page(self, path):
            name, ctype, needs_login = PAGES[path]
            authed = self._authed()
            if needs_login and not authed:
                return self._redirect("/login")
            if path == "/login" and auth.enabled and authed:
                return self._redirect("/")
            try:
                with open(os.path.join(STATIC_DIR, name), "rb") as f:
                    body = f.read()
            except OSError:
                return self._json(404, {"error": "missing " + name})
            extra = [("Content-Security-Policy", CSP)] if ctype.startswith("text/html") else []
            self._send(200, ctype, body, extra, cache="no-cache")

        def _info(self):
            self._json(200, {"name": ctx.name, "version": __version__, "login": auth.enabled,
                             "uptime": round(time.time() - ctx.started), "now": time.time(),
                             "storage": store.stats(), "debug": shared.debug,
                             "samples": ctx.samples.count() if ctx.samples else None})

        def _samples(self):
            if not ctx.samples:
                return self._json(404, {"error": "samples are switched off"})
            self._json(200, {"count": ctx.samples.count(), "latest": ctx.samples.latest(20)})

        def _sample_file(self, filename):
            path = ctx.samples.path(filename) if ctx.samples else None
            if not path:
                return self._json(404, {"error": "no such sample"})
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError:
                return self._json(404, {"error": "no such sample"})
            ctype = "image/jpeg" if filename.endswith(".jpg") else "application/json; charset=utf-8"
            self._send(200, ctype, body, cache="private, max-age=86400")

        def _history(self, query):
            days = _int(query, "days", 14, 1, 90)
            state = shared.get_state()
            self._json(200, {"days": store.history(days, open_stay=_open_stay(state)),
                             "zones": state.get("zones", [])})

        def _day(self, query):
            text = query.get("date", [""])[0]
            try:
                date = (datetime.datetime.strptime(text, "%Y-%m-%d").date() if text
                        else datetime.date.today())
            except ValueError:
                return self._json(400, {"error": "date must look like 2026-11-16"})
            self._json(200, store.day_detail(date, open_stay=_open_stay(shared.get_state())))

        def _snapshot(self, event_id):
            path = store.snapshot_path(event_id)
            if not path:
                return self._json(404, {"error": "no picture"})
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError:
                return self._json(404, {"error": "no picture"})
            self._send(200, "image/jpeg", body, cache="private, max-age=86400")

        def _frame(self, query):
            extra = []
            if query.get("raw", ["0"])[0] == "1":
                jpeg = shared.get_raw(max_age=3.0, timeout=2.5)
                watch = shared.raw_watch
                if jpeg is not None and watch:  # the editor shows what the detector looks at
                    extra.append(("X-Watch", ",".join("%.4f" % v for v in watch)))
            else:
                jpeg = shared.get_image(max_age=3.0, timeout=2.5)
            if jpeg is None:
                return self._json(503, {"error": "no picture from the camera yet"})
            self._send(200, "image/jpeg", jpeg, extra)

        def _export(self, query):
            days = _int(query, "days", 365, 1, 3650)
            now = time.time()
            events = store.events_between(now - days * 86400, now + 1, limit=200000)
            label = {"appear": "出现", "lost": "离开视野", "enter": "进入", "leave": "离开"}
            out = io.StringIO()
            out.write("﻿")  # so that Excel reads the Chinese right
            w = csv.writer(out)
            w.writerow(["时间", "事件", "区域", "停留秒数"])
            for e in events:
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["t"])),
                            label.get(e["type"], e["type"]), e["zone"] or "", e.get("duration", "")])
            self._send(200, "text/csv; charset=utf-8", out.getvalue().encode("utf-8"),
                       [("Content-Disposition", 'attachment; filename="catcam-events.csv"')])

        def _stream(self):
            if not shared.add_viewer():
                return self._json(503, {"error": "too many people are watching"})
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                seq, idle = 0, 0
                while True:
                    seq2, jpeg = shared.wait_frame(seq, 5.0)
                    if jpeg is None:
                        idle += 1
                        if idle >= 6:  # 30 s without a picture: let the browser reconnect
                            break
                        continue
                    idle, seq = 0, seq2
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                pass
            finally:
                shared.remove_viewer()

        # ---- POST

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/login", "/api/logout", "/api/zones", "/api/debug", "/api/sample"):
                return self._json(404, {"error": "not found"})
            if not self._same_origin():
                return self._json(403, {"error": "wrong origin"})
            if path == "/api/logout":
                return self._json(200, {"ok": True}, [("Set-Cookie", self._cookie_header("", 0))])
            body = self._body()
            if body is None:
                return
            if path == "/api/login":
                return self._login(body)
            if not self._authed():
                return self._json(401, {"error": "login required"})
            if path == "/api/debug":
                shared.debug = bool(body.get("on"))
                return self._json(200, {"ok": True, "debug": shared.debug})
            if path == "/api/sample":
                return self._keep_sample()
            self._save_zones(body)

        def _keep_sample(self):
            if not ctx.samples:
                return self._json(404, {"error": "samples are switched off"})
            name = shared.request_sample(timeout=4.0)
            if not name:
                return self._json(503, {"error": "没有摄像头画面，没能保存"})
            self._json(200, {"ok": True, "name": name, "count": ctx.samples.count()})

        def _cookie_header(self, value, max_age):
            return "%s=%s; Path=/; Max-Age=%d; HttpOnly; SameSite=Lax" % (auth.COOKIE, value, max_age)

        def _login(self, body):
            ok, wait = auth.login_allowed(self._ip())
            if not ok:
                return self._json(429, {"error": "试得太多了，请 %d 秒后再试" % wait},
                                  [("Retry-After", str(wait))])
            code = body.get("code")
            if auth.check_code(code.strip() if isinstance(code, str) else code):
                auth.record_success(self._ip())
                return self._json(200, {"ok": True},
                                  [("Set-Cookie", self._cookie_header(auth.new_cookie(), auth.MAX_AGE))])
            time.sleep(0.4)  # slows guessing down a little more
            auth.record_failure(self._ip())
            self._json(401, {"error": "口令不对"})

        def _save_zones(self, body):
            try:
                zones = parse_zones(body.get("zones"))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            data = [z.to_json() for z in zones]
            try:
                ctx.config.save_zones(data)
            except (OSError, ValueError) as exc:
                return self._json(500, {"error": "保存失败：%s" % exc})
            shared.zones_json = data
            shared.request_zones(zones)  # the main loop switches over before its next frame
            self._json(200, {"ok": True, "zones": data})

    return Handler


def serve(ctx, port, host="0.0.0.0"):
    server = _Server((host, port), _make_handler(ctx))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
