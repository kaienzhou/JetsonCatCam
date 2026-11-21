"use strict";
(function () {
  const form = document.getElementById("f");
  const input = document.getElementById("code");
  const msg = document.getElementById("msg");

  async function login(code) {
    msg.textContent = "";
    let res, data = {};
    try {
      res = await fetch("/api/login", {
        method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({code: code})
      });
      data = await res.json().catch(function () { return {}; });
    } catch (e) {
      msg.textContent = "连不上设备，请检查网络";
      return;
    }
    if (res.ok) { location.replace("/"); return; }
    msg.textContent = data.error || "登录失败";
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    login(input.value.trim());
  });

  // A link like /login#code=abcd... logs in by itself. The part after # stays in the browser;
  // it is never sent to the server in the request itself.
  const m = /(?:^|[#&])code=([^&]+)/.exec(location.hash);
  if (m) {
    history.replaceState(null, "", location.pathname);
    input.value = decodeURIComponent(m[1]);
    login(input.value);
  }
})();
