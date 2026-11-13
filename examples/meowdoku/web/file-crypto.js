// Client-side file encryption. Private keys live in memory only while this page is open.
// The server stores a passphrase-encrypted private key so a new device can unlock it.
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });
let privateKey = null, publicKey = null, fingerprint = null;

const bytes = (value) => new Uint8Array(value);
const b64 = (value) => {
  let text = "";
  const data = bytes(value);
  for (let i = 0; i < data.length; i += 8192) text += String.fromCharCode(...data.subarray(i, i + 8192));
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};
const unb64 = (value) => {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]+$/.test(value)) throw new Error("Invalid encrypted file metadata.");
  const raw = atob(value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4));
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
};
const random = (count) => crypto.getRandomValues(new Uint8Array(count));
const hex = (data) => Array.from(bytes(data), (x) => x.toString(16).padStart(2, "0")).join("");
const sha = async (data) => hex(await crypto.subtle.digest("SHA-256", data));
async function json(path, init = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", redirect: "error", ...init });
  let result;
  try { result = await response.json(); } catch (error) { throw new Error("Invalid server response."); }
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
async function passwordKey(passphrase, salt, iterations = 600000) {
  const base = await crypto.subtle.importKey("raw", encoder.encode(passphrase), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey({ name: "PBKDF2", hash: "SHA-256", salt, iterations }, base,
    { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
}
const rsa = { name: "RSA-OAEP", hash: "SHA-256" };
export const unlocked = () => !!privateKey;
export const status = () => json("/api/file-keys/me");

export async function setup(passphrase) {
  if (typeof passphrase !== "string" || passphrase.length < 12) throw new Error("Use a file recovery passphrase of at least 12 characters.");
  const prior = await status();
  if (prior.configured) throw new Error("A file key already exists. Unlock it with your recovery passphrase.");
  const pair = await crypto.subtle.generateKey({ name: "RSA-OAEP", modulusLength: 3072,
    publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" }, true, ["encrypt", "decrypt"]);
  const spki = await crypto.subtle.exportKey("spki", pair.publicKey);
  const pkcs8 = await crypto.subtle.exportKey("pkcs8", pair.privateKey);
  const salt = random(16), iv = random(12);
  const protectedKey = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, await passwordKey(passphrase, salt), pkcs8);
  const bundle = { public_spki: b64(spki), encrypted_private: b64(protectedKey), salt: b64(salt),
    iv: b64(iv), iterations: 600000 };
  const receipt = await json("/api/file-keys/me", { method: "POST",
    headers: { "Content-Type": "application/json", "X-Meowdoku-Key": "1" }, body: JSON.stringify(bundle) });
  privateKey = await crypto.subtle.importKey("pkcs8", pkcs8, rsa, false, ["decrypt"]);
  publicKey = pair.publicKey; fingerprint = await sha(spki);
  if (receipt.fingerprint !== fingerprint) throw new Error("File key fingerprint mismatch.");
  return fingerprint;
}

export async function unlock(passphrase) {
  const bundle = await status();
  if (!bundle.configured) throw new Error("Create your file key on the Keys page first.");
  try {
    const spki = unb64(bundle.public_spki);
    const raw = await crypto.subtle.decrypt({ name: "AES-GCM", iv: unb64(bundle.iv) },
      await passwordKey(passphrase, unb64(bundle.salt), bundle.iterations), unb64(bundle.encrypted_private));
    const secret = await crypto.subtle.importKey("pkcs8", raw, rsa, false, ["decrypt"]);
    const pub = await crypto.subtle.importKey("spki", spki, rsa, false, ["encrypt"]);
    const challenge = random(32);
    const wrapped = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, pub, challenge);
    const unwrapped = bytes(await crypto.subtle.decrypt({ name: "RSA-OAEP" }, secret, wrapped));
    if (!challenge.every((x, i) => x === unwrapped[i]) || await sha(spki) !== bundle.fingerprint)
      throw new Error("key mismatch");
    privateKey = secret; publicKey = pub; fingerprint = bundle.fingerprint;
    return fingerprint;
  } catch (error) { throw new Error("Wrong recovery passphrase or damaged file key."); }
}

function requireKey() {
  if (!privateKey) throw new Error("Unlock file encryption on the Keys page or in this chat first.");
}

async function recipientKey(username) {
  const peer = await json("/api/file-keys/peer?username=" + encodeURIComponent(username));
  const spki = unb64(peer.public_spki);
  if (await sha(spki) !== peer.fingerprint) throw new Error("Contact file key fingerprint mismatch.");
  const pin = "meowdoku-file-key:" + peer.username.toLowerCase();
  const known = localStorage.getItem(pin);
  if (known && known !== peer.fingerprint) throw new Error("Contact file key changed. Verify it with them before sending.");
  if (!known) {
    const verified = window.confirm("Compare " + peer.username + "'s file-key fingerprint with them through a channel outside Meowdoku:\n" +
      peer.fingerprint + "\n\nChoose OK only after it matches.");
    if (!verified) throw new Error("Recipient file key was not verified.");
    localStorage.setItem(pin, peer.fingerprint);
  }
  return { key: await crypto.subtle.importKey("spki", spki, rsa, false, ["encrypt"]), ...peer };
}

function safeName(name) {
  if (typeof name !== "string" || !name || name.length > 120 || /[/\\:\x00-\x1f\x7f]/.test(name))
    throw new Error("Choose a file with a simple name under 120 characters.");
  return name;
}

export async function seal(name, contents, recipient = null) {
  requireKey();
  name = safeName(name);
  const raw = bytes(contents);
  if (raw.length < 1 || raw.length > 8 * 1024 * 1024) throw new Error("Choose a file from 1 byte to 8 MiB.");
  const peer = recipient ? await recipientKey(recipient) : null;
  const fileKeyBytes = random(32);
  const fileKey = await crypto.subtle.importKey("raw", fileKeyBytes, "AES-GCM", false, ["encrypt", "decrypt"]);
  const iv = random(12), nameIv = random(12);
  const [cipher, nameCipher, ownerWrap, recipientWrap] = await Promise.all([
    crypto.subtle.encrypt({ name: "AES-GCM", iv }, fileKey, raw),
    crypto.subtle.encrypt({ name: "AES-GCM", iv: nameIv }, fileKey, encoder.encode(name)),
    crypto.subtle.encrypt({ name: "RSA-OAEP" }, publicKey, fileKeyBytes),
    peer ? crypto.subtle.encrypt({ name: "RSA-OAEP" }, peer.key, fileKeyBytes) : Promise.resolve(null),
  ]);
  const envelope = { v: 1, iv: b64(iv), name_iv: b64(nameIv), name_cipher: b64(nameCipher),
    owner_wrap: b64(ownerWrap), owner_fp: fingerprint, recipient: peer?.username || null,
    recipient_wrap: recipientWrap ? b64(recipientWrap) : null, recipient_fp: peer?.fingerprint || null };
  return { ciphertext: cipher, envelope };
}

export async function upload(name, contents, recipient = null) {
  const sealed = await seal(name, contents, recipient);
  const meta = b64(encoder.encode(JSON.stringify(sealed.envelope)));
  const response = await fetch("/api/user-files", { method: "POST", credentials: "same-origin", cache: "no-store",
    redirect: "error", headers: { "Content-Type": "application/octet-stream", "X-Meowdoku-Upload": "1",
      "X-Meowdoku-File-Meta": meta }, body: sealed.ciphertext });
  let result;
  try { result = await response.json(); } catch (error) { throw new Error("File upload response was invalid."); }
  if (!response.ok) throw new Error(result.error || "File upload failed.");
  if (!result || !/^[0-9a-f]{32}$/.test(result.id)) throw new Error("File upload receipt was invalid.");
  return { id: result.id, name };
}

async function fileKey(envelope) {
  requireKey();
  if (!envelope || envelope.v !== 1) throw new Error("Unsupported encrypted file.");
  const wrapped = envelope.owner_fp === fingerprint ? envelope.owner_wrap :
    envelope.recipient_fp === fingerprint ? envelope.recipient_wrap : null;
  if (!wrapped) throw new Error("This file was not encrypted for your key.");
  const raw = await crypto.subtle.decrypt({ name: "RSA-OAEP" }, privateKey, unb64(wrapped));
  return crypto.subtle.importKey("raw", raw, "AES-GCM", false, ["decrypt"]);
}

export async function decryptName(envelope) {
  const meta = typeof envelope === "string" ? JSON.parse(envelope) : envelope;
  return safeName(decoder.decode(await crypto.subtle.decrypt({ name: "AES-GCM", iv: unb64(meta.name_iv) },
    await fileKey(meta), unb64(meta.name_cipher))));
}

export async function openFile(id) {
  if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid file reference.");
  const response = await fetch("/api/user-files/" + id, { credentials: "same-origin", cache: "no-store", redirect: "error" });
  if (!response.ok) throw new Error("File is unavailable or no longer shared.");
  const header = response.headers.get("X-Meowdoku-File-Meta");
  const meta = JSON.parse(decoder.decode(unb64(header)));
  const key = await fileKey(meta);
  const [name, data] = await Promise.all([
    decryptName(meta), crypto.subtle.decrypt({ name: "AES-GCM", iv: unb64(meta.iv) }, key, await response.arrayBuffer()),
  ]);
  return { name, data };
}

export async function download(id) {
  const file = await openFile(id);
  const url = URL.createObjectURL(new Blob([file.data], { type: "application/octet-stream" }));
  const link = document.createElement("a"); link.href = url; link.download = file.name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return file.name;
}
