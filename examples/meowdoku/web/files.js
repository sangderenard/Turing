import * as fileCrypto from "/file-crypto.js";
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
    let fileName = "Encrypted file";
    if (fileCrypto.unlocked()) {
      try { fileName = await fileCrypto.decryptName(file.envelope); }
      catch (error) { fileName = "Encrypted file (key unavailable)"; }
    }
    const link = node("a", fileName);
    link.href = "/api/user-files/" + encodeURIComponent(file.id);
    link.addEventListener("click", async (event) => {
      event.preventDefault();
      try { status("Decrypting file…"); const name = await fileCrypto.download(file.id); status(`Downloaded ${name}.`); }
      catch (error) { status(error.message); }
    });
    line.append(link, node("small", `${(file.bytes / 1024).toFixed(1)} KiB`));
    if (fileName === "alpaca-paper-key.json" || fileName === "secret.json") {
      const openKeys = node("a", "Open on Keys page");
      openKeys.href = "/keys.html?import=" + encodeURIComponent(file.id);
      line.append(openKeys);
    }
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Share ${fileName} with a contact`);
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
          const original = await fileCrypto.openFile(file.id);
          const encrypted = await fileCrypto.upload(original.name, original.data, contact.username);
          await post("/api/messages/send", {
            body: "Shared an encrypted file",
            subject: { kind: "file", ref: encrypted.id },
            conversation_handle: contact.conversation_handle,
            idempotency_key: crypto.randomUUID(),
          });
          status(`Sent ${original.name} to ${contact.username}.`);
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
    const response = await fileCrypto.upload(file.name, await file.arrayBuffer());
    $("upload-form").reset();
    status(`Stored ${response.name} privately.`);
    await refresh();
  } catch (error) { status(error.message); }
  finally { button.disabled = false; }
});

$("unlock-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await fileCrypto.unlock($("unlock-passphrase").value);
    $("unlock-form").reset();
    status("File key unlocked for this page.");
    await refresh();
  } catch (error) { status(error.message); }
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
    await refresh();
  }
} catch (error) { status(error.message); }
