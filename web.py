"""Tiny stdlib web UI: MJPEG stream, JSON state, one static page.

No Flask: JetPack 4.6 ships Python 3.6 and we want zero pip installs.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CatCam</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
main{max-width:720px;margin:0 auto;padding:16px}
img{width:100%;border-radius:8px;background:#222}
.now{font-size:1.4rem;margin:12px 0}
li{margin:4px 0;color:#bbb}
h2{font-size:1rem;color:#888;margin-top:20px}
.fps{font-size:.8rem;color:#777;margin-top:6px;min-height:1em}
</style></head><body><main>
<img src="/stream.mjpg" alt="">
<div class="fps" id="fps"></div>
<div class="now" id="now">...</div>
<h2>今天各区域停留</h2><ul id="totals"></ul>
<h2>最近事件</h2><ul id="events"></ul>
</main>
<script>
const L={appear:"出现",lost:"离开视野",enter:"进入",leave:"离开"};
function hm(t){return new Date(t*1000).toLocaleTimeString()}
async function tick(){
  try{
    const s=await (await fetch("/api/state")).json();
    let text;
    if(s.status==="starting") text="启动中（首次运行要构建推理引擎，需要几分钟）…";
    else if(s.status==="no_camera") text="没有摄像头画面，请检查连接";
    else if(!s.present) text="猫不在画面里";
    else text=s.zone?("猫在：" + s.zone):"猫在画面里（不在任何区域）";
    document.getElementById("now").textContent=text;
    const t=s.timing;
    document.getElementById("fps").textContent=s.fps?(s.fps+" 帧/秒"+(t?
      "（采集 "+t.capture+" / 检测 "+t.detect+" / 转换 "+t.convert+" 毫秒）":"")):"";
    document.getElementById("totals").innerHTML=Object.entries(s.totals)
      .map(([z,v])=>"<li>"+z+"："+Math.round(v/60)+" 分钟</li>").join("")||"<li>暂无</li>";
    document.getElementById("events").innerHTML=s.events.slice().reverse().slice(0,30)
      .map(e=>"<li>"+hm(e.t)+" "+L[e.type]+(e.zone?" "+e.zone:"")+
        (e.duration!==undefined?"（"+e.duration+" 秒）":"")+"</li>").join("");
  }catch(e){}
  setTimeout(tick,1000);
}
tick();
</script></body></html>"""


class Shared(object):
    """Latest JPEG + state, written by the main loop, read by HTTP threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.state = {"status": "starting", "present": False, "zone": None,
                      "totals": {}, "events": []}

    def set_frame(self, jpeg):
        with self.lock:
            self.jpeg = jpeg

    def set_state(self, state):
        with self.lock:
            self.state = state

    def get(self):
        with self.lock:
            return self.jpeg, self.state


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _handler(shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            elif self.path == "/api/state":
                _, state = shared.get()
                self._send(200, "application/json", json.dumps(state, ensure_ascii=False).encode("utf-8"))
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        jpeg, _ = shared.get()
                        if jpeg is not None and jpeg is not last:
                            last = jpeg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

    return Handler


def serve(shared, port):
    server = _Server(("0.0.0.0", port), _handler(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
