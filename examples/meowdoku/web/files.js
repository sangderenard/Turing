const $ = (id) => document.getElementById(id);
const status = (message) => { $("status").textContent = message; };
const node = (tag, text) => { const out = document.createElement(tag); out.textContent = text; return out; };

async function json(url, init = {}) {
  const response = await fetch(url, { credentials: "same-origin", cache: "no-store", ...init });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function post(url, data) {
  return json(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

async function upload(name, body) {
  return json("/api/user-files", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream", "X-Meowdoku-Upload": "1",
               "X-Meowdoku-Filename": encodeURIComponent(name) },
    body,
  });
}

async function refresh() {
  const { files } = await json("/api/user-files");
  const conversations = await json("/api/messages/conversations");
  const list = $("files");
  list.replaceChildren();
  if (!files.length) list.append(node("li", "No files yet."));
  for (const file of files) {
    const item = document.createElement("li");
    const line = document.createElement("div");
    line.className = "row";
    const link = node("a", file.name);
    link.href = "/api/user-files/" + encodeURIComponent(file.id);
    line.append(link, node("small", `${(file.bytes / 1024).toFixed(1)} KiB`));
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Share ${file.name} with a contact`);
    select.append(new Option("Choose a contact", ""));
    const share = node("button", "Send in Messages");
    share.type = "button";
    share.disabled = true;
    select.addEventListener("change", () => { share.disabled = !select.value; });
    try {
      for (const contact of conversations.contacts || []) select.append(new Option(contact.username, contact.username));
      share.addEventListener("click", async () => {
        share.disabled = true;
        try {
          const current = await json("/api/messages/conversations");
          const contact = (current.contacts || []).find((c) => c.username === select.value);
          if (!contact) throw new Error("This contact is no longer available.");
          await post("/api/messages/send", {
            body: `Shared file: ${file.name}`,
            subject: { kind: "file", ref: file.id },
            conversation_handle: contact.conversation_handle,
            idempotency_key: crypto.randomUUID(),
          });
          status(`Sent ${file.name} to ${contact.username}.`);
        } catch (error) { status(error.message); }
        finally { share.disabled = false; }
      });
    } catch (error) { status(error.message); }
    line.append(select, share);
    item.append(line);
    list.append(item);
  }
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("chosen-file").files?.[0];
  if (!file) return;
  const button = $("upload-button");
  button.disabled = true;
  status("Uploading…");
  try {
    const response = await upload(file.name, file);
    $("upload-form").reset();
    status(`Stored ${response.name} privately.`);
    await refresh();
  } catch (error) { status(error.message); }
  finally { button.disabled = false; }
});

$("paper-key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const keyId = $("paper-id").value.trim();
  const secret = $("paper-secret").value.trim();
  if (!keyId || !secret || /[\r\n]/.test(keyId + secret)) { status("Enter both paper key values without line breaks."); return; }
  const button = $("paper-key-button");
  button.disabled = true;
  status("Saving privately…");
  try {
    await json("/api/user-secrets/alpaca-paper", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Meowdoku-Secret": "1" },
      body: JSON.stringify({ key_id: keyId, secret_key: secret }),
    });
    $("paper-key-form").reset();
    $("paper-key-state").textContent = "Paper key configured for this account.";
    status("Paper key saved. Its value is not shown or available for download.");
  } catch (error) { status(error.message); }
  finally { button.disabled = false; }
});

try {
  const who = await json("/api/whoami");
  if (!who.signed_in) {
    $("identity").replaceChildren(node("span", "Please "));
    const link = node("a", "sign in"); link.href = "/login.html";
    $("identity").append(link, node("span", " to upload files."));
  } else {
    $("identity").textContent = `Signed in as ${who.username}.`;
    $("upload-box").hidden = false;
    $("files-box").hidden = false;
    const connection = await json("/api/user-secrets/alpaca-paper");
    $("paper-key-state").textContent = connection.configured ? "Paper key configured for this account." : "No paper key configured yet.";
    await refresh();
  }
} catch (error) { status(error.message); }
