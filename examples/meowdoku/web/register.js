// Alpha signup uses the existing account route with an explicit, versioned agreement.
// Invitation codes remain in the URL fragment, so the server never receives one in an address.
(function () {
  "use strict";
  var form = document.getElementById("code-form"), input = document.getElementById("code");
  var alpha = document.getElementById("alpha-form"), submit = document.getElementById("alpha-submit");
  var status = document.getElementById("status");
  alpha.addEventListener("submit", function (event) {
    event.preventDefault();
    if (document.getElementById("password").value !== document.getElementById("password2").value) {
      status.textContent = "The two passwords differ."; status.className = "bad"; return;
    }
    submit.disabled = true;
    status.textContent = "Creating your alpha account..."; status.className = "mut";
    fetch("/api/account/signup", { method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({
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
          status.textContent = "Your basic member account is ready. Continue to your profile or sign in later.";
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
