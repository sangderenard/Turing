// Public sign-in page: POST /api/account/login, GET /api/account/me, POST /api/account/logout.  No sign-up here.
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var form = $("form"), who = $("who"), status = $("status"), submit = $("submit");

  function say(text, cls) { status.textContent = text || ""; status.className = cls || "mut"; }

  function show(info) {
    var account = info && info.account;
    form.hidden = !!account;
    who.hidden = !account;
    if (account) { $("name").textContent = account.username; $("role").textContent = info.role || "member"; }
  }

  function post(path, body) {
    return fetch(path, { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { return { status: r.status, body: j, retry: r.headers.get("Retry-After") }; }); });
  }

  function refresh() {
    return fetch("/api/account/me", { credentials: "same-origin" })
      .then(function (r) { return r.json(); }).then(show)
      .catch(function () { say("could not reach the server", "bad"); });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submit.disabled = true; say("signing in...");
    post("/api/account/login", { username: $("username").value, password: $("password").value }).then(function (res) {
      $("password").value = "";
      if (res.status === 200) { say(""); return refresh(); }
      if (res.status === 429) {
        var secs = Number(res.retry || (res.body && res.body.retry_after) || 0);
        var mins = Math.ceil(secs / 60);
        return say("Too many attempts. Try again in " + (secs > 90 ? mins + " minutes" : secs + " seconds") + ".", "bad");
      }
      say((res.body && res.body.error) || "sign-in failed", "bad");
    }).catch(function () { say("could not reach the server", "bad"); })
      .then(function () { submit.disabled = false; });
  });

  $("signout").addEventListener("click", function () {
    post("/api/account/logout", {}).then(function () { say("signed out"); return refresh(); });
  });

  refresh();
})();
