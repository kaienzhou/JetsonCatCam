"""The web server end to end: a real server on a free port, talked to over HTTP."""
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam import web
from catcam.auth import Auth
from catcam.config import Config
from catcam.store import Store

SQ = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]


def ts(y, mo, d, h=0, mi=0, s=0):
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


class WebTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({"port": 0, "note": "keep me", "zones": []}, f)
        data = os.path.join(self.dir, "data")
        self.auth = Auth(data)
        self.store = Store(os.path.join(data, "catcam.db"), os.path.join(data, "snaps"))
        self.shared = web.Shared()
        self.server = web.serve(web.Context(self.shared, self.auth, self.store, Config(self.config_path),
                                            name="testcam"), 0, "127.0.0.1")
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def req(self, method, path, body=None, headers=None, cookie=None, raw=None):
        h = dict(headers or {})
        if cookie:
            h["Cookie"] = cookie
        data = raw
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            h.setdefault("Content-Type", "application/json")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, data, h)
            r = conn.getresponse()
            return r.status, dict((k.lower(), v) for k, v in r.getheaders()), r.read()
        finally:
            conn.close()

    def login(self):
        status, headers, _ = self.req("POST", "/api/login", {"code": self.auth.code})
        self.assertEqual(status, 200)
        return headers["set-cookie"].split(";")[0]


class AccessTest(WebTestCase):
    def test_hello_is_public_and_says_what_this_is(self):
        status, _, body = self.req("GET", "/api/hello")
        info = json.loads(body.decode("utf-8"))
        self.assertEqual((status, info["app"], info["api"], info["name"], info["login"]),
                         (200, "catcam", 1, "testcam", True))

    def test_private_things_need_the_code(self):
        for path in ("/api/state", "/api/info", "/api/history", "/api/day", "/api/zones",
                     "/api/export.csv", "/api/frame.jpg", "/stream.mjpg", "/snap/1.jpg"):
            self.assertEqual(self.req("GET", path)[0], 401, path)
        status, headers, _ = self.req("GET", "/")
        self.assertEqual((status, headers["location"]), (302, "/login"))
        self.assertEqual(self.req("POST", "/api/zones", {"zones": []})[0], 401)

    def test_the_login_page_and_the_apps_own_files_are_public(self):
        for path, ctype in (("/login", "text/html"), ("/login.js", "application/javascript"),
                            ("/app.js", "application/javascript"), ("/app.css", "text/css"),
                            ("/manifest.webmanifest", "application/manifest+json"),
                            ("/icon-192.png", "image/png"), ("/icon-512.png", "image/png"),
                            ("/apple-touch-icon.png", "image/png"), ("/favicon.ico", "image/png")):
            status, headers, body = self.req("GET", path)
            self.assertEqual(status, 200, path)
            self.assertTrue(headers["content-type"].startswith(ctype), path)
            self.assertTrue(body, path)

    def test_login_sets_a_cookie_that_opens_the_door(self):
        cookie = self.login()
        self.assertTrue(cookie.startswith("catcam_session="))
        _, headers, _ = self.req("POST", "/api/login", {"code": self.auth.code})
        self.assertIn("HttpOnly", headers["set-cookie"])
        self.assertIn("SameSite=Lax", headers["set-cookie"])
        self.assertEqual(self.req("GET", "/api/state", cookie=cookie)[0], 200)
        status, headers, body = self.req("GET", "/", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn("script-src 'self'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertIn(b"app.js", body)
        # somebody who is already logged in and opens the login page is sent on
        self.assertEqual(self.req("GET", "/login", cookie=cookie)[0], 302)

    def test_a_wrong_code_is_refused_and_guessing_is_stopped(self):
        self.auth.max_fail = 2
        status, _, body = self.req("POST", "/api/login", {"code": "nope"})
        self.assertEqual(status, 401)
        self.assertIn("口令", json.loads(body.decode("utf-8"))["error"])
        self.assertEqual(self.req("POST", "/api/login", {"code": "nope"})[0], 401)
        status, headers, _ = self.req("POST", "/api/login", {"code": self.auth.code})  # even the right one
        self.assertEqual(status, 429)
        self.assertIn("retry-after", headers)

    def test_junk_in_the_login_body_is_just_a_refusal(self):
        for body in ({}, {"code": None}, {"code": 5}, {"code": ["a"]}):
            self.assertEqual(self.req("POST", "/api/login", body)[0], 401, body)

    def test_the_bearer_header_works_for_scripts(self):
        self.assertEqual(self.req("GET", "/api/state", headers={"Authorization": "Bearer " + self.auth.code})[0], 200)
        self.assertEqual(self.req("GET", "/api/state", headers={"Authorization": "Bearer wrong"})[0], 401)

    def test_a_forged_cookie_is_refused(self):
        forged = "catcam_session=" + "a" * 32 + "." + "b" * 64
        self.assertEqual(self.req("GET", "/api/state", cookie=forged)[0], 401)

    def test_logout_clears_the_cookie(self):
        _, headers, _ = self.req("POST", "/api/logout", {})
        self.assertIn("Max-Age=0", headers["set-cookie"])

    def test_posts_from_another_site_are_refused(self):
        evil = {"Origin": "http://evil.example"}
        self.assertEqual(self.req("POST", "/api/login", {"code": self.auth.code}, headers=evil)[0], 403)
        own = {"Origin": "http://127.0.0.1:%d" % self.port}
        self.assertEqual(self.req("POST", "/api/login", {"code": self.auth.code}, headers=own)[0], 200)

    def test_posts_must_be_json_and_not_huge(self):
        self.assertEqual(self.req("POST", "/api/login", raw=b"code=x",
                                  headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("POST", "/api/login", raw=b"{not json",
                                  headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.req("POST", "/api/login", raw=b"[1]",
                                  headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.req("POST", "/api/login", raw=b"x" * 70000,
                                  headers={"Content-Type": "application/json"})[0], 413)

    def test_unknown_paths(self):
        cookie = self.login()
        self.assertEqual(self.req("GET", "/nothing", cookie=cookie)[0], 404)
        self.assertEqual(self.req("POST", "/api/state", {}, cookie=cookie)[0], 404)
        self.assertEqual(self.req("GET", "/../../etc/passwd", cookie=cookie)[0], 404)

    def test_switched_off_login_lets_everybody_in(self):
        self.auth.enabled = False
        self.assertEqual(self.req("GET", "/api/state")[0], 200)
        self.assertEqual(self.req("GET", "/")[0], 200)


class DataTest(WebTestCase):
    def setUp(self):
        super(DataTest, self).setUp()
        self.cookie = self.login()

    def get_json(self, path):
        status, _, body = self.req("GET", path, cookie=self.cookie)
        self.assertEqual(status, 200, path)
        return json.loads(body.decode("utf-8"))

    def test_state_is_what_the_main_loop_published(self):
        self.shared.set_state({"status": "ok", "zone": "沙发", "events": []})
        self.assertEqual(self.get_json("/api/state"), {"status": "ok", "zone": "沙发", "events": []})

    def test_info(self):
        info = self.get_json("/api/info")
        self.assertEqual((info["name"], info["login"]), ("testcam", True))
        self.assertEqual(info["storage"]["events"], 0)

    def test_history_counts_the_stay_in_progress(self):
        self.shared.set_state({"zone": "沙发", "since": time.time() - 30, "zones": ["沙发"]})
        data = self.get_json("/api/history?days=3")
        self.assertEqual(len(data["days"]), 3)
        self.assertEqual(data["days"][-1]["date"], time.strftime("%Y-%m-%d"))
        self.assertGreater(data["days"][-1]["zones"]["沙发"]["seconds"], 0)
        self.assertEqual(data["zones"], ["沙发"])
        self.assertEqual(len(self.get_json("/api/history?days=nonsense")["days"]), 14)  # falls back
        self.assertEqual(len(self.get_json("/api/history?days=9999")["days"]), 90)      # capped

    def test_day_detail_and_pictures(self):
        enter = ts(2026, 11, 16, 9, 0)
        a = self.store.add({"t": enter, "type": "enter", "zone": "沙发"}, b"\xff\xd8picture")
        b = self.store.add({"t": enter + 600, "type": "leave", "zone": "沙发", "duration": 600.0})
        day = self.get_json("/api/day?date=2026-11-16")
        self.assertEqual(day["date"], "2026-11-16")
        self.assertEqual(day["hours"]["沙发"][9], 600.0)
        self.assertEqual([e["type"] for e in day["events"]], ["enter", "leave"])
        self.assertTrue(day["events"][0]["snap"])
        status, headers, body = self.req("GET", "/snap/%d.jpg" % a, cookie=self.cookie)
        self.assertEqual((status, headers["content-type"], body), (200, "image/jpeg", b"\xff\xd8picture"))
        self.assertEqual(self.req("GET", "/snap/%d.jpg" % b, cookie=self.cookie)[0], 404)  # no picture
        self.assertEqual(self.req("GET", "/snap/99999.jpg", cookie=self.cookie)[0], 404)
        self.assertEqual(self.req("GET", "/api/day?date=16.11.2026", cookie=self.cookie)[0], 400)

    def test_csv_export(self):
        self.store.add({"t": time.time() - 100, "type": "leave", "zone": "沙发", "duration": 12.5})
        status, headers, body = self.req("GET", "/api/export.csv?days=1", cookie=self.cookie)
        self.assertEqual(status, 200)
        self.assertIn("text/csv", headers["content-type"])
        self.assertIn("attachment", headers["content-disposition"])
        text = body.decode("utf-8")
        self.assertTrue(text.startswith("﻿时间,事件,区域,停留秒数"))
        self.assertIn("离开,沙发,12.5", text)

    def test_zones_get_and_save(self):
        self.shared.zones_json = [{"name": "旧", "points": SQ}]
        self.assertEqual(self.get_json("/api/zones"), {"zones": [{"name": "旧", "points": SQ}]})
        status, _, body = self.req("POST", "/api/zones", {"zones": [{"name": " 沙发 ", "points": SQ}]},
                                   cookie=self.cookie)
        self.assertEqual(status, 200, body)
        saved = json.loads(body.decode("utf-8"))["zones"]
        self.assertEqual(saved, [{"name": "沙发", "points": SQ}])
        with open(self.config_path, encoding="utf-8") as f:  # on disk, and the rest of the file is intact
            cfg = json.load(f)
        self.assertEqual((cfg["zones"], cfg["note"]), (saved, "keep me"))
        self.assertEqual(self.get_json("/api/zones")["zones"], saved)
        self.assertEqual([z.name for z in self.shared.take_zone_request()], ["沙发"])  # for the main loop
        self.assertIsNone(self.shared.take_zone_request())

    def read_config_text(self):
        with open(self.config_path, encoding="utf-8") as f:
            return f.read()

    def test_bad_zones_are_refused_with_a_reason_and_nothing_changes(self):
        before = self.read_config_text()
        status, _, body = self.req("POST", "/api/zones", {"zones": [{"name": "a", "points": SQ[:2]}]},
                                   cookie=self.cookie)
        self.assertEqual(status, 400)
        self.assertIn("3 到", json.loads(body.decode("utf-8"))["error"])
        self.assertEqual(self.req("POST", "/api/zones", {"nope": 1}, cookie=self.cookie)[0], 400)
        self.assertEqual(self.read_config_text(), before)
        self.assertIsNone(self.shared.take_zone_request())

    def test_a_config_that_cannot_be_written_is_reported(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("{ broken")
        status, _, body = self.req("POST", "/api/zones", {"zones": [{"name": "a", "points": SQ}]},
                                   cookie=self.cookie)
        self.assertEqual(status, 500)
        self.assertIn("保存失败", json.loads(body.decode("utf-8"))["error"])
        self.assertIsNone(self.shared.take_zone_request())


class PictureTest(WebTestCase):
    def setUp(self):
        super(PictureTest, self).setUp()
        self.cookie = self.login()

    def test_a_still_picture_asks_the_camera_loop_to_make_one(self):
        self.shared.set_frame(b"\xff\xd8annotated")
        status, headers, body = self.req("GET", "/api/frame.jpg", cookie=self.cookie)
        self.assertEqual((status, headers["content-type"], body), (200, "image/jpeg", b"\xff\xd8annotated"))
        self.assertTrue(self.shared.wants_frame())

    def test_the_editor_picture_has_no_drawings_and_switches_the_camera_loop_on(self):
        self.shared.set_raw(b"\xff\xd8plain")
        self.assertFalse(self.shared.wants_raw())
        status, _, body = self.req("GET", "/api/frame.jpg?raw=1", cookie=self.cookie)
        self.assertEqual((status, body), (200, b"\xff\xd8plain"))
        self.assertTrue(self.shared.wants_raw())

    def test_no_picture_yet(self):
        started = time.time()
        self.assertEqual(self.req("GET", "/api/frame.jpg?raw=1", cookie=self.cookie)[0], 503)
        self.assertLess(time.time() - started, 6)

    def test_the_stream_sends_multipart_jpegs_and_counts_viewers(self):
        self.shared.set_frame(b"\xff\xd8first")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/stream.mjpg", headers={"Cookie": self.cookie})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertIn("multipart/x-mixed-replace", resp.getheader("Content-Type"))
        chunk = resp.read(60)
        self.assertIn(b"--frame", chunk)
        self.assertEqual(self.shared.viewers, 1)
        self.assertTrue(self.shared.wants_frame())
        resp.close()  # the response owns the socket of a stream without Content-Length
        conn.close()
        for i in range(40):  # the handler notices the closed connection when it next writes
            self.shared.set_frame(b"\xff\xd8more%d" % i)
            if self.shared.viewers == 0:
                break
            time.sleep(0.05)
        self.assertEqual(self.shared.viewers, 0)

    def test_too_many_viewers(self):
        self.shared.viewers = web.MAX_VIEWERS
        self.assertEqual(self.req("GET", "/stream.mjpg", cookie=self.cookie)[0], 503)


if __name__ == "__main__":
    unittest.main()
