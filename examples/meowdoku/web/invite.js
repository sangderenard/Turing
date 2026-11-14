// MOCKUP of the invite page. Any signed-in account makes invitation links for people it knows; they can invite in turn.
// Live, a code is minted and recorded by the server (who invited whom), and an email goes out through the Nodus email
// manager table. Here the code is made in the browser and kept only until the page is left.
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var made = [];

  function render() {
    var list = $("list");
    list.textContent = "";
    if (!made.length) {
      var row = list.insertRow(), cell = row.insertCell();
      cell.colSpan = 3; cell.className = "mut"; cell.textContent = "No invitations yet.";
      return;
    }
    made.forEach(function (item) {
      var row = list.insertRow();
      row.insertCell().textContent = item.name + (item.email ? " (" + item.email + ")" : "");
      row.insertCell().textContent = item.at.toLocaleString();
      row.insertCell().textContent = "waiting (mock, not saved)";
    });
  }

  function newCode() {
    var bytes = new Uint8Array(9);
    crypto.getRandomValues(bytes);
    return Array.prototype.map.call(bytes, function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
  }

  fetch("/api/whoami", { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (who) {
    $("signed-out").hidden = !!who.signed_in;
    $("form").hidden = !who.signed_in;
  }).catch(function () { $("signed-out").hidden = false; });

  $("form").addEventListener("submit", function (event) {
    event.preventDefault();
    var item = { name: $("name").value.trim(), email: $("email").value.trim(), code: newCode(), at: new Date() };
    made.unshift(item);
    $("link").textContent = location.origin + "/invited.html?code=" + item.code;
    $("sent").textContent = item.email
      ? "Mockup: this would be emailed to " + item.email + " by the Nodus email manager, which is not connected yet."
      : "Send them this link yourself.";
    $("made").hidden = false;
    $("form").reset();
    render();
  });

  $("copy").addEventListener("click", function () {
    navigator.clipboard.writeText($("link").textContent).then(function () { $("copy").textContent = "Copied"; },
      function () { $("copy").textContent = "Select the link and copy it"; });
  });

  render();
})();
