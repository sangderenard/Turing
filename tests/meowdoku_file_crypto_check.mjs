import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const source = pathToFileURL(resolve("examples/meowdoku/web/file-crypto.js")).href;
const alice = await import(source + "?alice");
const bob = await import(source + "?bob");
const mallory = await import(source + "?mallory");
const fresh = await import(source + "?fresh");
const bundles = new Map(), files = new Map();
let actor = "alice", nextFile = 1;
const pinned = new Map();
globalThis.localStorage = { getItem: (key) => pinned.get(key) || null,
  setItem: (key, value) => pinned.set(key, value) };
globalThis.window = { confirm: () => true };
globalThis.fetch = async (path, init = {}) => {
  if (path === "/api/file-keys/me") {
    if (init.method === "POST") {
      const bundle = JSON.parse(init.body);
      assert.equal(bundles.has(actor), false);
      const raw = Buffer.from(bundle.public_spki.replace(/-/g, "+").replace(/_/g, "/"), "base64");
      const fingerprint = Buffer.from(await crypto.subtle.digest("SHA-256", raw)).toString("hex");
      bundles.set(actor, { configured: true, ...bundle, fingerprint });
      return Response.json({ configured: true, fingerprint }, { status: 201 });
    }
    return Response.json(bundles.get(actor) || { configured: false });
  }
  if (path.startsWith("/api/file-keys/peer?")) {
    const username = new URL("https://test.invalid" + path).searchParams.get("username");
    const key = bundles.get(username);
    return key ? Response.json({ username, public_spki: key.public_spki, fingerprint: key.fingerprint }) :
      Response.json({ error: "no key" }, { status: 404 });
  }
  if (path === "/api/user-files" && init.method === "POST") {
    const id = (nextFile++).toString(16).padStart(32, "0");
    files.set(id, { cipher: Buffer.from(init.body), meta: init.headers["X-Meowdoku-File-Meta"] });
    return Response.json({ id }, { status: 201 });
  }
  if (path.startsWith("/api/user-files/")) {
    const file = files.get(path.split("/").at(-1));
    return file ? new Response(file.cipher, { headers: { "X-Meowdoku-File-Meta": file.meta } }) :
      Response.json({ error: "missing" }, { status: 404 });
  }
  throw new Error("unexpected request " + path);
};

actor = "alice"; await alice.setup("long unique alice recovery phrase");
actor = "bob"; await bob.setup("long unique bob recovery phrase");
actor = "mallory"; await mallory.setup("long unique mallory recovery phrase");
actor = "alice";
const secret = new TextEncoder().encode('{"type":"alpaca-paper-key-v1","issuer":"Alpaca","purpose":"paper trading","key_id":"paper-example","secret_key":"never-print-this"}');
const uploaded = await alice.upload("alpaca-paper-key.json", secret, "bob");
const stored = files.get(uploaded.id);
assert.equal(stored.cipher.includes(Buffer.from("never-print-this")), false);
actor = "bob";
const opened = await bob.openFile(uploaded.id);
assert.equal(opened.name, "alpaca-paper-key.json");
assert.deepEqual(new Uint8Array(opened.data), secret);
actor = "alice";
assert.deepEqual(new Uint8Array((await alice.openFile(uploaded.id)).data), secret);
actor = "mallory";
await assert.rejects(() => mallory.openFile(uploaded.id), /not encrypted for your key/);
actor = "alice";
await assert.rejects(() => fresh.unlock("wrong passphrase"), /Wrong recovery passphrase/);
stored.cipher[0] ^= 1;
actor = "bob";
await assert.rejects(() => bob.openFile(uploaded.id));
console.log("file crypto: sender/recipient roundtrip, wrong passphrase, and tamper rejection passed");
