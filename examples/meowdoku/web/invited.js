// "You've been invited" (server: invitations.py): /invited.html#code=CODE. The code is read from the fragment (never sent
// to a server by the browser), taken out of the address bar at once, checked, and used once to register and sign in.
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  function say(text, cls) { $("status").textContent = text || ""; $("status").className = cls || "mut"; }
  function post(path, body) {
    return fetch(path, { method: "POST", credentials: "same-origin", cache: "no-store",
                         headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (b) { b.status = r.status; return b; }); });
  }

  var hash = new URLSearchParams(location.hash.slice(1)), query = new URLSearchParams(location.search);
  var code = (hash.get("code") || query.get("code") || "").trim();
  if (hash.has("code") || query.has("code")) history.replaceState(null, "", location.pathname);   // the code leaves the address bar

  if (!code) {
    $("lead").textContent = "This link has no invitation code. Ask the person who invited you for the whole link, or paste it on the Register page.";
    return;
  }
  post("/api/invitations/check", { code: code }).then(function (answer) {
    if (answer.status !== 200) { $("lead").textContent = answer.error || "This invitation could not be checked."; $("lead").className = "bad"; return; }
    $("lead").textContent = answer.inviter + " invited you. Choose a username and password to finish registering.";
    $("chain").textContent = "Invitation chain: " + answer.chain.slice().reverse().join(" → ") + " → you. " +
      "The link works until " + new Date(answer.expires * 1000).toLocaleString() + ".";
    $("finish").hidden = false;
    $("username").focus();
  }, function () { $("lead").textContent = "The server could not be reached. Try the link again in a moment."; });

  $("form").addEventListener("submit", function (event) {
    event.preventDefault();
    if ($("password").value !== $("password2").value) return say("The two passwords differ.", "bad");
    $("submit").disabled = true;
    say("Registering...");
    post("/api/invitations/redeem", { code: code, username: $("username").value.trim(), password: $("password").value })
      .then(function (answer) {
        $("submit").disabled = false;
        if (answer.status !== 201) return say(answer.error || "Not registered.", "bad");
        $("password").value = ""; $("password2").value = "";
        $("finish").hidden = true;
        $("lead").textContent = "Welcome, " + answer.account.username + ". You are registered and signed in.";
        say("Next: your profile, or the home page.", "ok");
        var links = document.createElement("p");
        links.innerHTML = '<a href="/profile.html">Your profile</a> · <a href="/">Home</a>';
        $("status").after(links);
      }, function () { $("submit").disabled = false; say("The server could not be reached. Try again.", "bad"); });
  });
})();
