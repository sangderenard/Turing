// Alpha signup uses the existing account route with an explicit, versioned agreement.
// Invitation codes remain in the URL fragment, so the server never receives one in an address.
(function () {
  "use strict";
  var form = document.getElementById("code-form"), input = document.getElementById("code");
  var alpha = document.getElementById("alpha-form"), submit = document.getElementById("alpha-submit");
  var status = document.getElementById("status");
  var previewPanel = document.getElementById("preview-panel");
  var previewStart = document.getElementById("preview-start");
  var previewExpiry = document.getElementById("preview-expiry");
  var temporary = false;

  function showAccount(account) {
    temporary = !!(account && account.temporary);
    if (previewStart) previewStart.hidden = !!account;
    if (previewExpiry) {
      previewExpiry.hidden = !temporary;
      if (temporary) previewExpiry.textContent = "Preview expires " + new Date(account.expires_at * 1000).toLocaleString() + ". Keep it below before then.";
    }
    if (temporary) {
      document.getElementById("alpha-heading").textContent = "Keep this account";
      submit.textContent = "Keep account and change password";
      status.textContent = "To keep this preview, choose a username and a unique password, then accept the terms below.";
      status.className = "mut";
    } else if (account) {
      alpha.hidden = true;
      if (previewPanel) previewPanel.hidden = true;
    }
  }
  fetch("/api/account/me", { credentials: "same-origin", cache: "no-store" })
    .then(function (response) { return response.json(); })
    .then(function (body) { showAccount(body.account); })
    .catch(function () { /* The form still works if the account check is unavailable. */ });

  if (previewStart) previewStart.addEventListener("click", function () {
    previewStart.disabled = true;
    status.textContent = "Starting your read-only preview..."; status.className = "mut";
    fetch("/api/account/preview", { method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json", "X-Meowdoku-Preview": "1" }, body: "{}"
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) throw new Error(body.error || "Could not start preview.");
        showAccount(body.account);
        status.textContent = "Your read-only preview is ready. Browse the site, or keep the account below.";
        status.className = "ok";
      });
    }).catch(function (error) { status.textContent = error.message; status.className = "bad"; })
      .finally(function () { previewStart.disabled = false; });
  });
  alpha.addEventListener("submit", function (event) {
    event.preventDefault();
    if (document.getElementById("password").value !== document.getElementById("password2").value) {
      status.textContent = "The two passwords differ."; status.className = "bad"; return;
    }
    submit.disabled = true;
    status.textContent = temporary ? "Keeping your account..." : "Creating your alpha account..."; status.className = "mut";
    fetch(temporary ? "/api/account/keep" : "/api/account/signup", { method: "POST", credentials: "same-origin", cache: "no-store",
      headers: temporary ? { "Content-Type": "application/json", "X-Meowdoku-Preview": "1" } : { "Content-Type": "application/json" }, body: JSON.stringify({
        username: document.getElementById("username").value.trim(),
        password: document.getElementById("password").value,
        alpha_terms_version: alpha.dataset.termsVersion,
        agree_mit: document.getElementById("agree-mit").checked,
        agree_observation: document.getElementById("agree-observation").checked,
        agree_research: document.getElementById("agree-research").checked
      }) }).then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (body) {
          if (!response.ok) throw new Error(body.error || "Alpha registration failed.");
          document.getElementById("password").value = "";
          document.getElementById("password2").value = "";
          alpha.hidden = true;
          if (previewPanel) previewPanel.hidden = true;
          status.textContent = "Your basic member account is ready. Your new password is set; use it to sign in on another browser.";
          status.className = "ok";
          var next = document.createElement("p");
          var link = document.createElement("a");
          link.href = "/profile.html"; link.textContent = "Open your profile";
          next.appendChild(link); status.after(next);
        });
      }).catch(function (error) { status.textContent = error.message; status.className = "bad"; })
        .finally(function () { submit.disabled = false; });
  });
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = input.value.trim(), code = text;
    try {
      var url = new URL(text, location.origin);
      code = new URLSearchParams(url.hash.slice(1)).get("code") || url.searchParams.get("code") || text;
    } catch (e) { code = text; }
    input.value = "";
    location.assign("/invited.html#code=" + encodeURIComponent(code));
  });
})();
