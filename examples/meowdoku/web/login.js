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
    if (account) serverPanel(); else $("server").hidden = true;
  }

  // Stable server / Development server: two shared instances of the whole service (not the dev page vs shader page, not a per-user server).
  // Shown only when whoami lists the dev_ui capability.  POST /api/route {site} stores the choice for this account and sets the route cookie.
  function serverPanel() {
    var box = $("server");
    Promise.all([fetch("/api/whoami", { credentials: "same-origin" }).then(function (r) { return r.json(); }),
                 fetch("/api/route", { credentials: "same-origin" }).then(function (r) { return r.ok ? r.json() : {}; })])
      .then(function (both) {
        var who = both[0] || {}, route = both[1] || {};
        var allowed = (who.capabilities || []).indexOf("dev_ui") >= 0;
        box.hidden = !allowed;
        if (!allowed) return;
        $("instance").textContent = who.instance || "unnamed";
        var dev = route.dev_route && route.selected === route.dev_route;
        var chosen = route.preference || (dev ? "dev" : "stable");
        $("site-stable").setAttribute("aria-pressed", String(chosen === "stable"));
        $("site-dev").setAttribute("aria-pressed", String(chosen === "dev"));
        $("site-dev").disabled = route.dev_available === false;
        $("server-note").textContent = route.dev_available === false
          ? "The Development server is not enabled; you stay on the Stable server."
          : "Your choice is remembered for this account.";
      }).catch(function () { box.hidden = true; });
  }

  function chooseServer(site) {
    post("/api/route", { site: site }).then(function (res) {
      if (res.status === 200) return location.reload();
      say((res.body && res.body.error) || "could not switch server", "bad");
    });
  }
  $("site-stable").addEventListener("click", function () { chooseServer("stable"); });
  $("site-dev").addEventListener("click", function () { chooseServer("dev"); });

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
