"use strict";
(function () {
  const view = document.getElementById("view");
  const connEl = document.getElementById("conn");
  const PALETTE = ["#f5a623", "#4fc3f7", "#b388ff", "#3ecf8e", "#f48fb1", "#ff8a65", "#90caf9", "#e57373"];
  const TYPE_LABEL = {appear: "出现", lost: "离开视野", enter: "进入", leave: "离开"};
  const SVGNS = "http://www.w3.org/2000/svg";
  const MAX_ZONES = 12, MAX_POINTS = 24;

  // ---------- tiny DOM helpers (text always goes in as text, never as HTML)

  function fill(node, props, kids) {
    Object.keys(props || {}).forEach(function (key) {
      const v = props[key];
      if (v === null || v === undefined || v === false) return;
      if (key === "text") node.textContent = v;
      else if (key.slice(0, 2) === "on") node.addEventListener(key.slice(2), v);
      else node.setAttribute(key, v === true ? "" : String(v));
    });
    kids.reduce(function (all, k) { return all.concat(k); }, []).forEach(function (kid) {
      if (kid === null || kid === undefined || kid === false) return;
      node.appendChild(kid.nodeType ? kid : document.createTextNode(String(kid)));
    });
    return node;
  }
  function h(tag, props) {
    return fill(document.createElement(tag), props, Array.prototype.slice.call(arguments, 2));
  }
  function svg(tag, props) {
    return fill(document.createElementNS(SVGNS, tag), props, Array.prototype.slice.call(arguments, 2));
  }
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // ---------- formatting

  function pad(n) { return n < 10 ? "0" + n : "" + n; }
  function hm(t) {
    const d = new Date(t * 1000);
    return pad(d.getHours()) + ":" + pad(d.getMinutes());
  }
  function dur(sec) {
    sec = Math.max(0, Math.round(sec));
    if (sec < 60) return sec + " 秒";
    const m = Math.round(sec / 60);
    if (m < 60) return m + " 分钟";
    const hours = Math.floor(m / 60), rest = m % 60;
    return hours + " 小时" + (rest ? " " + rest + " 分" : "");
  }
  function mb(bytes) {
    return bytes < 1048576 ? Math.round(bytes / 1024) + " KB" : (bytes / 1048576).toFixed(1) + " MB";
  }

  let zoneOrder = [];
  function colorOf(name) {
    let i = zoneOrder.indexOf(name);
    if (i < 0) {
      let hash = 0;
      for (let k = 0; k < name.length; k++) hash = (hash * 31 + name.charCodeAt(k)) >>> 0;
      i = hash;
    }
    return PALETTE[i % PALETTE.length];
  }

  // ---------- talking to the camera

  function setConn(ok) {
    connEl.textContent = ok ? "" : "连接断开";
    connEl.className = "conn" + (ok ? "" : " bad");
  }
  async function api(path, opts) {
    let res;
    try {
      res = await fetch(path, Object.assign({credentials: "same-origin", cache: "no-store"}, opts));
    } catch (e) {
      setConn(false);
      throw new Error("连不上相机，请检查网络");
    }
    setConn(true);
    if (res.status === 401) {
      location.href = "/login";
      throw new Error("请先登录");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) throw new Error((data && data.error) || "出错了（" + res.status + "）");
    return data;
  }
  function postJson(path, body) {
    return api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  }

  let toastTimer = null;
  function toast(text, bad) {
    let el = document.getElementById("toast");
    if (!el) {
      el = h("div", {id: "toast", role: "status"});
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.className = "toast show" + (bad ? " bad" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.classList.remove("show"); }, 2600);
  }

  function showPicture(url, caption) {
    const overlay = h("div", {class: "overlay", onclick: function () { overlay.remove(); }},
      h("img", {src: url, alt: ""}), h("div", {class: "caption", text: caption}));
    document.body.appendChild(overlay);
  }

  function describe(e) {
    return (TYPE_LABEL[e.type] || e.type) + (e.zone ? " " + e.zone : "") +
      (e.duration !== undefined ? "（" + dur(e.duration) + "）" : "");
  }
  function eventRow(e) {
    const kids = [h("span", {class: "when", text: hm(e.t)}), h("span", {class: "grow", text: describe(e)})];
    if (e.snap) {
      const url = "/snap/" + e.id + ".jpg";
      kids.push(h("img", {class: "thumb", loading: "lazy", alt: "", src: url,
        onclick: function () { showPicture(url, hm(e.t) + " " + describe(e)); }}));
    }
    return h("div", {class: "ev"}, kids);
  }

  // ---------- live

  const Live = {
    alive: false,
    gen: 0,
    start: function () {
      this.alive = true;
      this.img = h("img", {class: "stream", alt: "实时画面", onerror: () => this.retry()});
      this.status = h("div", {class: "status"});
      this.sub = h("div", {class: "muted small"});
      this.totals = h("div", {class: "bars"});
      this.events = h("div");
      view.appendChild(h("div", {class: "card nopad"}, h("div", {class: "frame"}, this.img)));
      view.appendChild(h("div", {class: "card"}, this.status, this.sub));
      view.appendChild(h("div", {class: "card"}, h("h2", {text: "今天各区域停留"}), this.totals));
      view.appendChild(h("div", {class: "card"}, h("h2", {text: "最近事件"}), this.events));
      this.play();
      this.tick(++this.gen);
    },
    stop: function () {
      this.alive = false;
      this.gen++;
      clearTimeout(this.timer);
      clearTimeout(this.retryTimer);
      if (this.img) this.img.removeAttribute("src");
    },
    visibility: function (visible) {  // no stream while the phone is locked or the tab is hidden
      if (!this.alive) return;
      clearTimeout(this.timer);
      this.gen++;
      if (visible) { this.play(); this.tick(this.gen); } else this.img.removeAttribute("src");
    },
    play: function () { this.img.src = "/stream.mjpg?t=" + Date.now(); },
    retry: function () {
      clearTimeout(this.retryTimer);
      this.retryTimer = setTimeout(() => { if (this.alive && !document.hidden) this.play(); }, 2000);
    },
    tick: async function (gen) {
      if (!this.alive || gen !== this.gen) return;
      try {
        const s = await api("/api/state");
        if (this.alive && gen === this.gen) this.render(s);
      } catch (e) { /* the connection indicator says it */ }
      if (this.alive && gen === this.gen) this.timer = setTimeout(() => this.tick(gen), 1000);
    },
    render: function (s) {
      if (s.zones) zoneOrder = s.zones;
      let main, sub = "";
      if (s.status === "starting") main = "启动中……首次运行要构建推理引擎，需要几分钟";
      else if (s.status === "no_camera") main = "没有摄像头画面，请检查连接";
      else if (!s.present) main = "猫不在画面里";
      else if (s.zone) { main = "猫在：" + s.zone; sub = "已经 " + dur(s.now - s.since); }
      else main = "猫在画面里（不在任何区域）";
      this.status.textContent = main;
      this.status.className = "status" + (s.zone ? " in" : "");
      const t = s.timing;
      this.sub.textContent = sub || (s.fps ? s.fps + " 帧/秒" + (t && t.detect ? "（检测 " + t.detect + " 毫秒）" : "") : "");

      clear(this.totals);
      const entries = Object.keys(s.totals || {}).map(function (k) { return [k, s.totals[k]]; })
        .sort(function (a, b) { return b[1] - a[1]; });
      if (!entries.length) this.totals.appendChild(h("div", {class: "muted", text: "今天还没有记录"}));
      const max = Math.max.apply(null, entries.map(function (e) { return e[1]; }).concat([1]));
      entries.forEach((e) => {
        const n = (s.visits || {})[e[0]] || 0;
        this.totals.appendChild(h("div", {class: "brow"},
          h("div", {class: "row between"},
            h("span", null, h("i", {class: "dot", style: "background:" + colorOf(e[0])}), e[0]),
            h("span", {class: "muted", text: dur(e[1]) + (n ? " · " + n + " 次" : "")})),
          h("div", {class: "bar"}, h("span", {style: "width:" + Math.max(2, e[1] / max * 100) + "%;background:" + colorOf(e[0])}))));
      });

      clear(this.events);
      const evs = (s.events || []).slice(-8).reverse();
      if (!evs.length) this.events.appendChild(h("div", {class: "muted", text: "还没有事件"}));
      evs.forEach((e) => this.events.appendChild(eventRow(e)));
    }
  };

  // ---------- history

  const History = {
    alive: false,
    days: 14,
    selected: null,
    data: null,
    start: function () {
      this.alive = true;
      this.chart = h("div");
      this.detail = h("div");
      const seg = h("div", {class: "seg"});
      [7, 14, 30].forEach((n) => {
        seg.appendChild(h("button", {class: n === this.days ? "on" : "", text: n + " 天", onclick: (ev) => {
          this.days = n;
          Array.prototype.forEach.call(seg.children, function (b) { b.classList.remove("on"); });
          ev.target.classList.add("on");
          this.load();
        }}));
      });
      view.appendChild(h("div", {class: "card"},
        h("div", {class: "row between"}, h("h2", {text: "每天停留时间"}), seg), this.chart));
      view.appendChild(this.detail);
      this.load();
    },
    stop: function () { this.alive = false; },
    load: async function () {
      clear(this.chart);
      this.chart.appendChild(h("div", {class: "muted", text: "加载中……"}));
      try {
        this.data = await api("/api/history?days=" + this.days);
      } catch (e) {
        clear(this.chart);
        this.chart.appendChild(h("div", {class: "msg", text: e.message}));
        return;
      }
      if (!this.alive) return;
      if (this.data.zones && this.data.zones.length) zoneOrder = this.data.zones;
      const dates = this.data.days.map(function (d) { return d.date; });
      if (!this.selected || dates.indexOf(this.selected) < 0) this.selected = dates[dates.length - 1];
      this.drawChart();
      this.showDay(this.selected);
    },
    drawChart: function () {
      const days = this.data.days;
      const totals = days.map(function (d) {
        return Object.keys(d.zones).reduce(function (sum, z) { return sum + d.zones[z].seconds; }, 0);
      });
      const max = Math.max.apply(null, totals.concat([60]));
      const seen = {};
      const chart = h("div", {class: "chart"});
      days.forEach((d, i) => {
        const stack = h("div", {class: "stack", style: "height:" + (totals[i] / max * 118) + "px"});
        Object.keys(d.zones).sort().forEach(function (z) {
          const sec = d.zones[z].seconds;
          seen[z] = true;
          if (sec > 0) stack.appendChild(h("div", {class: "seg-part",
            style: "height:" + (sec / totals[i] * 100) + "%;background:" + colorOf(z)}));
        });
        const show = days.length <= 14 || (days.length - 1 - i) % 5 === 0;
        chart.appendChild(h("button", {class: "col" + (d.date === this.selected ? " sel" : ""), title: d.date,
          onclick: () => { this.selected = d.date; this.drawChart(); this.showDay(d.date); }},
          totals[i] > 0 ? stack : h("div", {class: "empty"}),
          h("div", {class: "lab", text: show ? d.date.slice(5).replace("-", "/") : " "})));
      });
      clear(this.chart);
      this.chart.appendChild(chart);
      const names = Object.keys(seen).sort();
      if (names.length) {
        this.chart.appendChild(h("div", {class: "legend"}, names.map(function (n) {
          return h("span", null, h("i", {class: "dot", style: "background:" + colorOf(n)}), n);
        })));
      } else {
        this.chart.appendChild(h("p", {class: "muted", text: "这段时间还没有记录。猫进入某个区域后，这里就会出现统计。"}));
      }
    },
    showDay: async function (date) {
      clear(this.detail);
      const card = h("div", {class: "card"}, h("div", {class: "muted", text: "加载中……"}));
      this.detail.appendChild(card);
      let data;
      try {
        data = await api("/api/day?date=" + date);
      } catch (e) {
        card.textContent = e.message;
        return;
      }
      if (!this.alive || this.selected !== date) return;
      clear(card);
      const d = new Date(date + "T00:00:00");
      card.appendChild(h("div", {class: "row between"},
        h("h2", {text: (d.getMonth() + 1) + " 月 " + d.getDate() + " 日 周" + "日一二三四五六".charAt(d.getDay())}),
        h("a", {class: "btn small", href: "/api/export.csv?days=" + Math.max(7, this.days),
          download: "catcam-events.csv", text: "导出 CSV"})));
      const summary = this.data.days.filter(function (x) { return x.date === date; })[0];
      const names = Object.keys(data.hours).sort();
      if (!names.length && !data.events.length) {
        card.appendChild(h("p", {class: "muted", text: "这一天没有记录。"}));
        return;
      }
      names.forEach(function (z) {
        const row = data.hours[z];
        const total = row.reduce(function (a, b) { return a + b; }, 0);
        const visits = summary && summary.zones[z] ? summary.zones[z].visits : 0;
        card.appendChild(h("div", {class: "zline"},
          h("i", {class: "dot", style: "background:" + colorOf(z)}), h("span", {class: "name", text: z}),
          h("span", {class: "muted", text: dur(total) + (visits ? " · " + visits + " 次" : "")})));
        card.appendChild(h("div", {class: "hours"}, row.map(function (sec, hour) {
          return h("i", {title: hour + ":00  " + dur(sec),
            style: sec > 0 ? "background:" + colorOf(z) + ";opacity:" + Math.max(0.3, Math.min(1, sec / 3600)) : null});
        })));
      });
      if (names.length) {
        card.appendChild(h("div", {class: "axis"}, ["0 点", "6 点", "12 点", "18 点", "24 点"].map(function (t) {
          return h("span", {text: t});
        })));
      }
      if (data.events.length) {
        card.appendChild(h("h2", {text: "这天的事件"}));
        data.events.slice().reverse().forEach(function (e) { card.appendChild(eventRow(e)); });
      }
    }
  };

  // ---------- zones: draw the areas on a still picture of the room

  function colorFor(i) { return PALETTE[i % PALETTE.length]; }
  function centroid(points) {
    const n = points.length;
    return [points.reduce(function (s, p) { return s + p[0]; }, 0) / n,
            points.reduce(function (s, p) { return s + p[1]; }, 0) / n];
  }
  function round4(x) { return Math.round(x * 10000) / 10000; }

  const Zones = {
    alive: false,
    zones: null,       // [{name, points: [[x, y], ...]}], 0..1 on both axes; kept while you visit other tabs
    sel: -1,
    drawing: null,     // points of the zone being drawn, or null
    dirty: false,
    ratio: 0.75,       // picture height / width
    drag: null,

    start: function () {
      this.alive = true;
      this.drawing = null;
      this.sel = -1;
      this.bg = h("img", {alt: ""});
      this.wait = h("div", {class: "wait", text: "正在取摄像头画面……"});
      this.svg = svg("svg", {viewBox: "0 0 1000 " + 1000 * this.ratio, preserveAspectRatio: "none"});
      this.stage = h("div", {class: "stage"}, this.bg, this.svg, this.wait);
      this.stage.style.setProperty("--ratio", this.ratio * 100 + "%");
      this.toolbar = h("div", {class: "row"});
      this.hint = h("p", {class: "muted small"});
      this.list = h("div");
      this.msg = h("p", {class: "msg"});
      this.saveBtn = h("button", {class: "btn primary", text: "保存区域", onclick: () => this.save()});
      view.appendChild(h("div", {class: "card"}, this.toolbar, h("div", {style: "height:10px"}), this.stage, this.hint));
      view.appendChild(h("div", {class: "card"}, h("h2", {text: "区域"}), this.list, this.msg,
        h("div", {class: "row", style: "margin-top:8px"}, this.saveBtn)));
      this.svg.addEventListener("pointerdown", (ev) => this.onDown(ev));
      this.svg.addEventListener("pointermove", (ev) => this.onMove(ev));
      this.svg.addEventListener("pointerup", (ev) => this.onUp(ev));
      this.svg.addEventListener("pointercancel", (ev) => this.onUp(ev));
      this.onResize = () => this.drawSvg();
      window.addEventListener("resize", this.onResize);
      this.renderAll();
      this.loadZones();
      clearTimeout(this.picTimer);
      this.refreshPicture();
    },
    stop: function () {
      this.alive = false;
      clearTimeout(this.picTimer);
      clearTimeout(this.press);
      window.removeEventListener("resize", this.onResize);
      this.drag = null;
    },
    loadZones: async function () {
      if (this.zones) return;
      try {
        const data = await api("/api/zones");
        this.zones = data.zones.map(function (z) {
          return {name: z.name, points: z.points.map(function (p) { return [p[0], p[1]]; })};
        });
        zoneOrder = this.zones.map(function (z) { return z.name; });
      } catch (e) {
        this.msg.textContent = e.message;
        return;
      }
      if (this.alive) this.renderAll();
    },
    setRatio: function (r) {
      if (Math.abs(r - this.ratio) < 0.001) return;
      this.ratio = r;
      this.stage.style.setProperty("--ratio", r * 100 + "%");
      this.svg.setAttribute("viewBox", "0 0 1000 " + 1000 * r);
      this.drawSvg();
    },
    refreshPicture: async function () {
      if (!this.alive) return;
      let delay = 1500;
      if (!document.hidden) {
        try {
          const res = await fetch("/api/frame.jpg?raw=1&t=" + Date.now(), {credentials: "same-origin", cache: "no-store"});
          if (res.status === 401) { location.href = "/login"; return; }
          if (!res.ok) throw new Error("no picture");
          const url = URL.createObjectURL(await res.blob());
          if (!this.alive) { URL.revokeObjectURL(url); return; }
          await new Promise((resolve, reject) => {
            const probe = new Image();
            probe.onload = () => { if (probe.naturalWidth) this.setRatio(probe.naturalHeight / probe.naturalWidth); resolve(); };
            probe.onerror = reject;
            probe.src = url;
          });
          if (this.lastUrl) URL.revokeObjectURL(this.lastUrl);
          this.lastUrl = url;
          this.bg.src = url;
          this.wait.style.display = "none";
        } catch (e) {
          if (!this.bg.getAttribute("src")) {
            this.wait.textContent = "还没有摄像头画面，稍等一下……";
            this.wait.style.display = "";
          }
          delay = 2500;
        }
      }
      if (this.alive) this.picTimer = setTimeout(() => this.refreshPicture(), delay);
    },

    // -- drawing the overlay

    drawSvg: function () {
      clear(this.svg);
      if (!this.zones) return;
      const W = 1000, H = 1000 * this.ratio;
      const k = W / (this.svg.getBoundingClientRect().width || W);  // CSS pixels -> viewBox units
      const pt = function (p) { return p[0] * W + "," + p[1] * H; };
      this.zones.forEach((z, i) => {
        const color = colorFor(i);
        this.svg.appendChild(svg("polygon", {points: z.points.map(pt).join(" "), fill: color,
          "fill-opacity": i === this.sel ? 0.3 : 0.17, stroke: color, "stroke-width": i === this.sel ? 3 : 2,
          "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke", "data-zone": i}));
        const c = centroid(z.points);
        this.svg.appendChild(svg("text", {x: c[0] * W, y: c[1] * H, "text-anchor": "middle",
          "dominant-baseline": "middle", "font-size": 15 * k, fill: "#fff", stroke: "rgba(0,0,0,.7)",
          "stroke-width": 3 * k, "paint-order": "stroke", "pointer-events": "none",
          text: z.name || "（没有名字）"}));
      });
      if (this.sel >= 0 && this.zones[this.sel] && !this.drawing) {
        const color = colorFor(this.sel);
        this.zones[this.sel].points.forEach((p, j) => {
          this.svg.appendChild(svg("circle", {cx: p[0] * W, cy: p[1] * H, r: 13 * k, fill: color, stroke: "#fff",
            "stroke-width": 2, "vector-effect": "non-scaling-stroke", "data-h": j}));
        });
      }
      if (this.drawing && this.drawing.length) {
        const color = colorFor(this.zones ? this.zones.length : 0);
        if (this.drawing.length > 1) {
          this.svg.appendChild(svg("polyline", {points: this.drawing.map(pt).join(" "), fill: "none", stroke: color,
            "stroke-width": 3, "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke", "pointer-events": "none"}));
        }
        this.drawing.forEach((p, j) => {
          this.svg.appendChild(svg("circle", {cx: p[0] * W, cy: p[1] * H, r: (j === 0 ? 15 : 10) * k, fill: color,
            stroke: "#fff", "stroke-width": 2, "vector-effect": "non-scaling-stroke", "pointer-events": "none"}));
        });
      }
    },
    renderToolbar: function () {
      clear(this.toolbar);
      if (this.drawing) {
        this.toolbar.appendChild(h("button", {class: "btn primary", text: "完成（" + this.drawing.length + " 个点）",
          disabled: this.drawing.length < 3, onclick: () => this.finishDrawing()}));
        this.toolbar.appendChild(h("button", {class: "btn", text: "撤销一点", disabled: !this.drawing.length,
          onclick: () => { this.drawing.pop(); this.drawSvg(); this.renderToolbar(); }}));
        this.toolbar.appendChild(h("button", {class: "btn", text: "取消",
          onclick: () => { this.drawing = null; this.renderAll(); }}));
        this.hint.textContent = "在画面里依次点出区域的几个角；点第一个点或按「完成」封口。";
      } else {
        this.toolbar.appendChild(h("button", {class: "btn primary", text: "＋ 新区域", onclick: () => {
          if (this.zones && this.zones.length >= MAX_ZONES) { toast("最多 " + MAX_ZONES + " 个区域", true); return; }
          this.drawing = [];
          this.sel = -1;
          this.renderAll();
        }}));
        this.hint.textContent = "点区域选中它：拖圆点改形状，拖区域里面整体移动，点边线加顶点，长按圆点删顶点。";
      }
    },
    renderList: function () {
      clear(this.list);
      if (!this.zones) return;
      if (!this.zones.length) {
        this.list.appendChild(h("p", {class: "muted", text: "还没有区域。点「＋ 新区域」在画面上圈一块，比如沙发、食盆。"}));
      }
      this.zones.forEach((z, i) => {
        const input = h("input", {value: z.name, maxlength: 24, placeholder: "区域名字", "aria-label": "区域名字",
          oninput: (ev) => { z.name = ev.target.value; this.touched(); },
          onfocus: () => { if (this.sel !== i) this.select(i); }});
        this.list.appendChild(h("div", {class: "zrow" + (i === this.sel ? " sel" : ""), "data-i": i,
          onclick: (ev) => { if (ev.target.tagName === "DIV") this.select(i); }},
          h("i", {class: "dot", style: "background:" + colorFor(i)}), input,
          h("button", {class: "btn small danger", text: "删除", onclick: () => {
            this.zones.splice(i, 1);
            this.sel = -1;
            this.dirty = true;
            this.renderAll();
          }})));
      });
    },
    markSelected: function () {
      Array.prototype.forEach.call(this.list.children, (row) => {
        row.classList.toggle("sel", row.getAttribute("data-i") === String(this.sel));
      });
    },
    updateSave: function () {
      this.saveBtn.disabled = !this.dirty;
      this.saveBtn.textContent = this.dirty ? "保存区域（有改动）" : "保存区域";
    },
    renderAll: function () {
      this.renderToolbar();
      this.renderList();
      this.drawSvg();
      this.updateSave();
    },
    select: function (i) {
      this.sel = i;
      this.drawSvg();
      this.markSelected();
    },
    touched: function () {
      this.dirty = true;
      this.msg.textContent = "";
      this.drawSvg();
      this.updateSave();
    },

    // -- touching the picture

    pos: function (ev) {
      const r = this.svg.getBoundingClientRect();
      return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
              Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
    },
    onDown: function (ev) {
      if (!this.zones) return;
      ev.preventDefault();
      const p = this.pos(ev);
      if (this.drawing) { this.addPoint(p); return; }
      const handle = ev.target.getAttribute("data-h");
      const zone = ev.target.getAttribute("data-zone");
      if (handle !== null && this.sel >= 0) { this.startVertexDrag(ev, parseInt(handle, 10)); return; }
      if (zone !== null) {
        const i = parseInt(zone, 10);
        if (i === this.sel && this.tryInsertVertex(ev, p)) return;
        if (i !== this.sel) this.select(i);
        this.drag = {kind: "zone", start: p, orig: this.zones[i].points.map(function (q) { return q.slice(); }), moved: false};
        this.svg.setPointerCapture(ev.pointerId);
        return;
      }
      if (this.sel >= 0 && this.tryInsertVertex(ev, p)) return;
      this.select(-1);
    },
    startVertexDrag: function (ev, j) {
      const z = this.zones[this.sel];
      this.drag = {kind: "vertex", j: j, startX: ev.clientX, startY: ev.clientY, moved: false};
      this.svg.setPointerCapture(ev.pointerId);
      clearTimeout(this.press);
      this.press = setTimeout(() => {  // a long press removes the corner
        if (!this.drag || this.drag.kind !== "vertex" || this.drag.moved) return;
        this.drag = null;
        if (z.points.length > 3) { z.points.splice(j, 1); this.touched(); toast("已删除一个顶点"); }
        else toast("区域至少要 3 个顶点", true);
      }, 600);
    },
    tryInsertVertex: function (ev, p) {
      const z = this.zones[this.sel];
      if (!z || z.points.length >= MAX_POINTS) return false;
      const r = this.svg.getBoundingClientRect();
      let best = null;
      for (let i = 0; i < z.points.length; i++) {
        const a = z.points[i], b = z.points[(i + 1) % z.points.length];
        const ax = a[0] * r.width, ay = a[1] * r.height, bx = b[0] * r.width, by = b[1] * r.height;
        const px = p[0] * r.width, py = p[1] * r.height;
        const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy || 1;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
        const qx = ax + t * dx, qy = ay + t * dy, dist = Math.hypot(px - qx, py - qy);
        if (!best || dist < best.dist) best = {i: i, dist: dist, q: [qx / r.width, qy / r.height]};
      }
      if (!best || best.dist > 14) return false;
      z.points.splice(best.i + 1, 0, best.q);
      this.drag = {kind: "vertex", j: best.i + 1, startX: ev.clientX, startY: ev.clientY, moved: true};
      this.svg.setPointerCapture(ev.pointerId);
      this.touched();
      return true;
    },
    onMove: function (ev) {
      const d = this.drag;
      if (!d || this.sel < 0) return;
      const p = this.pos(ev);
      const z = this.zones[this.sel];
      if (d.kind === "vertex") {
        if (!d.moved && Math.hypot(ev.clientX - d.startX, ev.clientY - d.startY) < 6) return;
        d.moved = true;
        clearTimeout(this.press);
        z.points[d.j] = p;
        this.touched();
      } else if (d.kind === "zone") {
        const r = this.svg.getBoundingClientRect();
        let dx = p[0] - d.start[0], dy = p[1] - d.start[1];
        if (!d.moved && Math.hypot(dx * r.width, dy * r.height) < 6) return;
        d.moved = true;
        const xs = d.orig.map(function (q) { return q[0]; }), ys = d.orig.map(function (q) { return q[1]; });
        dx = Math.max(-Math.min.apply(null, xs), Math.min(1 - Math.max.apply(null, xs), dx));
        dy = Math.max(-Math.min.apply(null, ys), Math.min(1 - Math.max.apply(null, ys), dy));
        z.points = d.orig.map(function (q) { return [q[0] + dx, q[1] + dy]; });
        this.touched();
      }
    },
    onUp: function (ev) {
      clearTimeout(this.press);
      this.drag = null;
      try { this.svg.releasePointerCapture(ev.pointerId); } catch (e) { /* not captured */ }
    },
    addPoint: function (p) {
      const pts = this.drawing;
      if (pts.length >= 3) {
        const r = this.svg.getBoundingClientRect();
        if (Math.hypot((p[0] - pts[0][0]) * r.width, (p[1] - pts[0][1]) * r.height) < 20) { this.finishDrawing(); return; }
      }
      if (pts.length >= MAX_POINTS) { toast("最多 " + MAX_POINTS + " 个顶点", true); return; }
      pts.push(p);
      this.drawSvg();
      this.renderToolbar();
    },
    freeName: function () {
      const used = this.zones.map(function (z) { return z.name; });
      let n = 1;
      while (used.indexOf("区域" + n) >= 0) n++;
      return "区域" + n;
    },
    finishDrawing: function () {
      if (!this.drawing || this.drawing.length < 3) { toast("至少要 3 个点", true); return; }
      this.zones.push({name: this.freeName(), points: this.drawing});
      this.sel = this.zones.length - 1;
      this.drawing = null;
      this.dirty = true;
      this.msg.textContent = "";
      this.renderAll();
      toast("给新区域起个名字，再点保存");
    },
    save: async function () {
      if (!this.zones) return;
      this.msg.textContent = "";
      const body = {zones: this.zones.map(function (z) {
        return {name: z.name.trim(), points: z.points.map(function (p) { return [round4(p[0]), round4(p[1])]; })};
      })};
      this.saveBtn.disabled = true;
      try {
        const res = await postJson("/api/zones", body);
        this.zones = res.zones.map(function (z) { return {name: z.name, points: z.points}; });
        zoneOrder = this.zones.map(function (z) { return z.name; });
        this.dirty = false;
        this.sel = -1;
        toast("已保存，马上生效");
      } catch (e) {
        this.msg.textContent = e.message;
        toast(e.message, true);
      }
      if (this.alive) this.renderAll();
    }
  };
  window.addEventListener("beforeunload", function (ev) {
    if (Zones.dirty) { ev.preventDefault(); ev.returnValue = ""; }
  });

  // ---------- settings

  const Settings = {
    alive: false,
    start: function () {
      this.alive = true;
      const card = h("div", {class: "card"}, h("div", {class: "muted", text: "加载中……"}));
      view.appendChild(card);
      this.load(card);
    },
    stop: function () { this.alive = false; },
    load: async function (card) {
      let info;
      try {
        info = await api("/api/info");
      } catch (e) {
        card.textContent = e.message;
        return;
      }
      if (!this.alive) return;
      const kv = function (k, v) { return h("div", {class: "kv"}, h("span", {text: k}), h("span", {text: v})); };
      clear(card);
      card.appendChild(h("h2", {text: "这台相机"}));
      [kv("名字", info.name), kv("版本", info.version), kv("已运行", dur(info.uptime)),
       kv("记录的事件", info.storage.events + " 条"),
       kv("事件图片", info.storage.snapshots + " 张 · " + mb(info.storage.snapshot_bytes)),
       kv("数据库", mb(info.storage.db_bytes)), kv("当前地址", location.origin)].forEach(function (row) {
        card.appendChild(row);
      });
      view.appendChild(h("div", {class: "card"}, h("h2", {text: "数据"}),
        h("div", {class: "row"}, h("a", {class: "btn", href: "/api/export.csv?days=3650",
          download: "catcam-events.csv", text: "导出全部记录（CSV）"}))));
      view.appendChild(h("div", {class: "card"}, h("h2", {text: "当成手机 App 用"}),
        h("p", {class: "small", text: "iPhone：用 Safari 打开本页 → 分享按钮 → 添加到主屏幕。"}),
        h("p", {class: "small", text: "安卓：用 Chrome 打开本页 → 菜单 → 添加到主屏幕。"}),
        h("p", {class: "muted small", text: "相机换了网络，地址（IP）可能会变。到时候在路由器里找到它，或用 USB 线连电脑查看。"})));
      if (info.login) {
        view.appendChild(h("div", {class: "card"}, h("button", {class: "btn danger", text: "退出登录", onclick: async function () {
          try { await postJson("/api/logout", {}); } catch (e) { /* leave anyway */ }
          location.href = "/login";
        }})));
      }
    }
  };

  // ---------- tabs

  const tabs = {live: Live, history: History, zones: Zones, settings: Settings};
  let current = null;
  function route() {
    let name = location.hash.replace(/^#\/?/, "") || "live";
    if (!tabs[name]) name = "live";
    if (current && current.stop) current.stop();
    Array.prototype.forEach.call(document.querySelectorAll("#tabs a"), function (a) {
      a.classList.toggle("on", a.getAttribute("data-tab") === name);
    });
    clear(view);
    window.scrollTo(0, 0);
    current = tabs[name];
    current.start();
  }
  window.addEventListener("hashchange", route);
  document.addEventListener("visibilitychange", function () {
    if (current && current.visibility) current.visibility(!document.hidden);
  });
  route();
})();
