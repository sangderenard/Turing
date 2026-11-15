// Registration is by invitation: the page turns a pasted invitation link or code into /invited.html#code=... (the code in
// the fragment, so no server receives it in an address).
(function () {
  "use strict";
  var form = document.getElementById("code-form"), input = document.getElementById("code");
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
