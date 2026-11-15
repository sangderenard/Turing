// The invite page (server: invitations.py). A signed-in account makes an invitation: a one-time code in a link and its QR
// code, shown once. The code is in the link's fragment (#code=...), which no server ever receives. Nothing is emailed from
// the server: "Email it" opens your own mail app with the link filled in, and "Share" uses the device's share sheet.
import { drawQr } from "/tipjar.js";

const $ = (id) => document.getElementById(id);
const when = (seconds) => (seconds == null ? "" : new Date(seconds * 1000).toLocaleString());

async function request(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  let result = {};
  try { result = await response.json(); } catch (error) { /* no body */ }
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function describe(item) {
  if (item.state === "used") return `used by ${item.used_by}, ${when(item.used_at)}`;
  if (item.state === "taken back") return `taken back ${when(item.revoked_at)}`;
  if (item.state === "expired") return `expired ${when(item.expires)}`;
  if (item.state === "registering") return "being used right now";
  return `waiting, until ${when(item.expires)}`;
}

async function render() {
  const list = $("list");
  let invitations = [];
  try { ({ invitations } = await request("/api/invitations")); } catch (error) { $("sent").textContent = error.message; return; }
  list.replaceChildren();
  if (!invitations.length) {
    const row = list.insertRow(), cell = row.insertCell();
    cell.colSpan = 4; cell.className = "mut"; cell.textContent = "No invitations yet.";
    return;
  }
  for (const item of invitations) {
    const row = list.insertRow();
    row.insertCell().textContent = item.note + (item.email ? ` (${item.email})` : "");
    row.insertCell().textContent = when(item.created);
    row.insertCell().textContent = describe(item);
    const act = row.insertCell();
    if (item.state === "waiting") {
      const back = document.createElement("button");
      back.type = "button"; back.textContent = "Take back"; back.className = "small";
      back.onclick = async () => {
        if (!confirm(`Take back the invitation for ${item.note}? Its link stops working.`)) return;
        try { await request(`/api/invitations/${item.id}/revoke`, {}); await render(); } catch (error) { alert(error.message); }
      };
      act.append(back);
    }
  }
}

function show(made, note, email) {
  const link = location.origin + made.path;
  $("link").textContent = link;
  $("code").textContent = made.code;
  $("until").textContent = `It works once, until ${when(made.invitation.expires)}. This is the only time the code is shown: ` +
    "copy it, send it or print this page now.";
  try { drawQr($("qr"), link, 264); $("qr").hidden = false; } catch (error) { $("qr").hidden = true; }
  const subject = "An invitation to join me on SpeakToMe";
  const text = `Hi ${note},\n\nI've invited you to SpeakToMe. Open this link to choose your username and password:\n\n${link}\n\n` +
    `It works once, until ${when(made.invitation.expires)}.`;
  $("mail").href = `mailto:${encodeURIComponent(email || "")}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(text)}`;
  $("share").hidden = !navigator.share;
  $("share").onclick = () => navigator.share({ title: subject, text, url: link }).catch(() => {});
  $("copy").textContent = "Copy link";
  $("copy").onclick = () => navigator.clipboard.writeText(link).then(() => { $("copy").textContent = "Copied"; },
    () => { $("copy").textContent = "Select the link and copy it"; });
  $("made").hidden = false;
  $("made").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function main() {
  let who = {};
  try { who = await request("/api/whoami"); } catch (error) { /* shown as signed out */ }
  $("signed-out").hidden = !!who.signed_in;
  $("form").hidden = !who.signed_in;
  $("yours").hidden = !who.signed_in;
  if (!who.signed_in) return;
  $("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const note = $("name").value.trim(), email = $("email").value.trim();
    $("make").disabled = true;
    try {
      const made = await request("/api/invitations", { note, email, days: Number($("days").value) });
      show(made, note, email);
      $("form").reset();
      await render();
    } catch (error) { $("sent").textContent = `Not made: ${error.message}`; }
    finally { $("make").disabled = false; }
  });
  $("print").onclick = () => window.print();
  await render();
}
main();
