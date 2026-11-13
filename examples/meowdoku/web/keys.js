import * as fileCrypto from "/file-crypto.js";
const $ = (id) => document.getElementById(id);
const say = (message) => { $("status").textContent = message; };
async function json(path, init = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", redirect: "error", ...init });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
const post = (path, body, extra = {}) => json(path, { method: "POST",
  headers: { "Content-Type": "application/json", ...extra }, body: JSON.stringify(body) });
let ownKey = null, imported = null;
const importId = new URLSearchParams(location.search).get("import");

async function refresh() {
  ownKey = await fileCrypto.status();
  $("setup-form").hidden = ownKey.configured;
  $("unlock-form").hidden = !ownKey.configured || fileCrypto.unlocked();
  $("file-key-state").textContent = ownKey.configured ? (fileCrypto.unlocked() ? "Unlocked for this page." : "Ready to unlock on this device.") : "Create a file key before sending encrypted files.";
  $("fingerprint").textContent = ownKey.fingerprint || "—";
  const paper = await json("/api/user-secrets/alpaca-paper");
  $("paper-state").textContent = paper.configured ? "Paper key connected to this account." : "No paper key connected yet.";
}

async function afterUnlock() { await refresh(); if (importId) await loadImport(importId); }
$("setup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("setup-pass").value !== $("setup-confirm").value) { say("Passphrases do not match."); return; }
  const button = $("setup-form").querySelector("button"); button.disabled = true;
  say("Generating and protecting your file key…");
  try { await fileCrypto.setup($("setup-pass").value); $("setup-form").reset(); say("File key created. Keep your recovery passphrase."); await afterUnlock(); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});
$("unlock-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await fileCrypto.unlock($("unlock-pass").value); $("unlock-form").reset(); say("File key unlocked for this page."); await afterUnlock(); }
  catch (error) { say(error.message); }
});

function paperValues() {
  const key_id = $("paper-id").value.trim(), secret_key = $("paper-secret").value.trim();
  if (!key_id || !secret_key || /[\r\n]/.test(key_id + secret_key)) throw new Error("Enter a paper key ID and secret without line breaks.");
  return { key_id, secret_key };
}
const paperFile = () => ({ type: "alpaca-paper-key-v1", issuer: "Alpaca", purpose: "paper trading", ...paperValues() });
$("paper-connect").addEventListener("click", async () => {
  const button = $("paper-connect"); button.disabled = true;
  try { await post("/api/user-secrets/alpaca-paper", paperValues(), { "X-Meowdoku-Secret": "1" });
    $("paper-form").reset(); say("Paper key connected to this account."); await refresh(); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});
$("paper-store").addEventListener("click", async () => {
  const button = $("paper-store"); button.disabled = true;
  try { const file = await fileCrypto.upload("alpaca-paper-key.json",
      new TextEncoder().encode(JSON.stringify(paperFile())));
    $("paper-form").reset(); say("Encrypted paper key stored privately. Open it from My files to connect or share it later (" + file.id.slice(0, 8) + "…)."); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});

async function sendEncrypted(name, content, recipient, body) {
  if (!recipient) throw new Error("Choose an accepted contact.");
  if (!fileCrypto.unlocked()) throw new Error("Unlock your file key first.");
  const contacts = await json("/api/messages/conversations");
  const contact = (contacts.contacts || []).find((c) => c.username === recipient);
  if (!contact) throw new Error("This contact is no longer available.");
  const file = await fileCrypto.upload(name, new TextEncoder().encode(JSON.stringify(content)), recipient);
  await post("/api/messages/send", { body, subject: { kind: "file", ref: file.id },
    conversation_handle: contact.conversation_handle, idempotency_key: crypto.randomUUID() });
  return file;
}
$("paper-share").addEventListener("click", async () => {
  const button = $("paper-share"); button.disabled = true;
  try { const content = paperFile(); const peer = $("paper-recipient").value;
    await sendEncrypted("alpaca-paper-key.json", content, peer,
      "Shared an encrypted Alpaca paper key");
    $("paper-form").reset(); say("Encrypted paper key sent to " + peer + "."); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});
$("general-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("general-form").querySelector("button"); button.disabled = true;
  try { const issuer = $("secret-issuer").value.trim(), purpose = $("secret-purpose").value.trim(), content = $("secret-content").value;
    if (!issuer || !purpose || !content) throw new Error("Enter an issuer, purpose and secret content.");
    const peer = $("general-recipient").value;
    await sendEncrypted("secret.json", { type: "external-secret-v1", issuer, purpose, content }, peer,
      "Shared an encrypted secret");
    $("general-form").reset(); say("Encrypted secret sent to " + peer + "."); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});
$("general-store").addEventListener("click", async () => {
  const button = $("general-store"); button.disabled = true;
  try { const issuer = $("secret-issuer").value.trim(), purpose = $("secret-purpose").value.trim(), content = $("secret-content").value;
    if (!issuer || !purpose || !content) throw new Error("Enter an issuer, purpose and secret content.");
    await fileCrypto.upload("secret.json", new TextEncoder().encode(JSON.stringify({ type: "external-secret-v1", issuer, purpose, content })));
    $("general-form").reset(); say("Encrypted secret stored privately. Share it from My files when ready."); }
  catch (error) { say(error.message); }
  finally { button.disabled = false; }
});

async function loadImport(id) {
  if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid shared file reference.");
  const file = await fileCrypto.openFile(id);
  let data;
  try { data = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(file.data)); }
  catch (error) { throw new Error("This encrypted file is not a supported secret file."); }
  imported = data;
  $("import-box").hidden = false;
  $("import-connect").hidden = data.type !== "alpaca-paper-key-v1";
  $("import-reveal").hidden = data.type !== "external-secret-v1";
  $("import-state").textContent = data.type === "alpaca-paper-key-v1" ?
    "Encrypted Alpaca-issued paper key received. Key ID ending " + String(data.key_id || "").slice(-4) + "." :
    data.type === "external-secret-v1" ? "Encrypted credential from " + String(data.issuer || "unknown") + " for " + String(data.purpose || "unknown") + "." : "Unsupported secret type.";
}
$("import-connect").addEventListener("click", async () => {
  if (imported?.type !== "alpaca-paper-key-v1") return;
  try { await post("/api/user-secrets/alpaca-paper", { key_id: imported.key_id, secret_key: imported.secret_key },
    { "X-Meowdoku-Secret": "1" });
    imported = null; $("import-connect").hidden = true; say("Shared paper key connected to your account."); await refresh(); }
  catch (error) { say(error.message); }
});
$("import-reveal").addEventListener("click", () => {
  if (imported?.type !== "external-secret-v1") return;
  $("import-content").value = imported.content;
  $("import-content").hidden = false;
});

try {
  const who = await json("/api/whoami");
  if (!who.signed_in) { $("identity").innerHTML = 'Please <a href="/login.html">sign in</a> to manage keys.'; }
  else {
    $("identity").textContent = "Signed in as " + who.username + ".";
    $("paper-box").hidden = false; $("general-box").hidden = false;
    const conversations = await json("/api/messages/conversations");
    for (const contact of conversations.contacts || []) {
      $("paper-recipient").append(new Option(contact.username, contact.username));
      $("general-recipient").append(new Option(contact.username, contact.username));
    }
    await refresh();
    if (importId && fileCrypto.unlocked()) await loadImport(importId);
    else if (importId) { $("import-box").hidden = false; $("import-state").textContent = "Unlock your file key to open this secret."; }
  }
} catch (error) { say(error.message); }
