// MOCKUP. Registration is closed: the page only turns an invitation code or link into /invited.html?code=...
(function () {
  "use strict";
  var form = document.getElementById("code-form"), input = document.getElementById("code");
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = input.value.trim(), code = text;
    try { code = new URL(text, location.origin).searchParams.get("code") || text; } catch (e) { code = text; }
    location.assign("/invited.html?code=" + encodeURIComponent(code));
  });
})();
