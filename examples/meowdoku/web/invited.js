// MOCKUP of the "you've been invited" landing: /invited.html?code=CODE.
// Live, it will look the code up (who invited, and who invited them) and finish registration with it. Nothing is sent now.
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var code = new URLSearchParams(location.search).get("code");
  function say(text, cls) { $("status").textContent = text || ""; $("status").className = cls || "mut"; }

  if (!code) {
    $("lead").textContent = "This link has no invitation code. Ask the person who invited you for the whole link, or enter the code on the Register page.";
    return;
  }
  // Mock lookup: a live server answers with the inviter and their chain.
  $("lead").textContent = "albert invited you. Choose a username and password to finish registering.";
  $("chain").textContent = "Invitation chain (mock): admin \u2192 albert \u2192 you. Code " + code + ".";
  $("finish").hidden = false;

  $("form").addEventListener("submit", function (event) {
    event.preventDefault();
    if ($("password").value !== $("password2").value) return say("The two passwords differ.", "bad");
    $("password").value = ""; $("password2").value = "";
    say("Mockup: invitations are not live yet, so no account was made.", "bad");
  });
})();
