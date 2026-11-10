import assert from "node:assert/strict";
import test from "node:test";
import {
  marbleDownloadFilename, createMarbleObjectDownload, createLivePngDownload,
  downloadMarbleObject, downloadLivePng,
} from "../examples/meowdoku/web/marble_downloads.js";

const timestamp = "2026-11-09T16:23:45.678Z";
const digest = "abcdef0123456789".repeat(4);
const snapshot = {
  v: 1, generator: "meowdoku-marble-object-v1", name: "Café blue / marble",
  design: {seed: 17, "base-shape": "marble", kind: "clear", "bubble-count": 0,
    mesh_lods: {medium: digest, low: digest, proxy: digest},
    analytic_spheres: {version: 1, spheres: [{center: [0, 0, 0], radius: 1, material: 1}]},
    opaqueSaved: null},
  materials: {index: {marble_clear_glass: 1}, extra: {coated: 2, paintA: 3, paintB: 4},
    tables: {pbr: {stride: 8, data: [0, 1, 0.25, 3, 4, 5, 6, 7]}}},
  physical: {"radius-mm": 10, "shell-density": 2500},
  derived: {massKg: 0.01047, inertia: [1, 2, 3]}, geometry_sha256: digest,
  future_extension: {keep: [true, false, "all authored fields"]},
};
// A complete 1x1 PNG. The helper must retain these exact bytes and Blob identity.
const pngBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=", "base64");
const png = new Blob([pngBytes], {type: "image/png"});

function browser({fail = ""} = {}) {
  const events = [], downloads = [], active = new Set();
  let sequence = 0;
  const document = {
    body: {appendChild(anchor) {
      events.push("append");
      if(fail === "append")throw new Error("append failed");
      anchor.connected = true;
    }},
    createElement(tag) {
      assert.equal(tag, "a");
      if(fail === "create")throw new Error("create failed");
      return {
        click() {
          assert.equal(this.connected, true);
          assert.equal(this.hidden, true);
          assert.equal(active.has(this.href), true);
          events.push("click");
          if(fail === "click")throw new Error("click failed");
          downloads.push({filename: this.download, url: this.href});
        },
        remove() {events.push("remove"); this.connected = false;},
      };
    },
  };
  const blobs = [];
  const URL = {
    createObjectURL(blob) {
      if(fail === "url")throw new Error("url failed");
      const url = `blob:test/${++sequence}`;
      active.add(url);blobs.push(blob);events.push("url");return url;
    },
    revokeObjectURL(url) {
      assert.equal(active.delete(url), true, "each export URL is revoked exactly once");
      events.push("revoke");
    },
  };
  const delay = async ms => {
    assert.equal(ms, 1000);assert.equal(active.size > 0, true);
    events.push("delay");
    if(fail === "delay")throw new Error("delay failed");
  };
  return {document, URL, delay, timestamp, digest, events, downloads, active, blobs};
}

test("JSON export preserves full versioned snapshot and UTF-8 bytes without mutation", async () => {
  const before = JSON.stringify(snapshot);
  const artifact = createMarbleObjectDownload(snapshot, {timestamp});
  assert.equal(artifact.filename, "marble-Cafe-blue-marble-object-20261109T162345678Z-abcdef012345.json");
  assert.equal(artifact.blob.type, "application/json;charset=utf-8");
  const bytes = Buffer.from(await artifact.blob.arrayBuffer());
  assert.deepEqual(bytes, Buffer.from(`${JSON.stringify(snapshot, null, 2)}\n`, "utf8"));
  assert.deepEqual(JSON.parse(bytes.toString("utf8")), snapshot);
  assert.equal(JSON.stringify(snapshot), before);
});

test("filenames cannot inject paths, controls, alternate extensions or device names", () => {
  for(const name of ["../../evil.exe", "C:\\Users\\user\\CON", "NUL", "..", "\0\r\n", "", "💎", "a".repeat(400), "x\u202Egnp.exe"]) {
    const filename = marbleDownloadFilename(name, {kind: "live", timestamp, digest});
    assert.match(filename, /^marble-[a-z0-9_-]+-live-20261109T162345678Z-abcdef012345\.png$/i);
    assert.equal(filename.length < 150, true);
    assert.equal(/[\\/:\x00-\x1f\u202e]/.test(filename), false);
  }
  assert.throws(() => marbleDownloadFilename("x", {kind: "../png", timestamp}), /kind/);
  assert.throws(() => marbleDownloadFilename("x", {kind: "live", timestamp: "bad"}), /timestamp/);
  assert.throws(() => marbleDownloadFilename("x", {kind: "live", timestamp, digest: "../../abc"}), /digest/);
});

test("JSON export refuses unsupported recipes and lossy or unserializable values", () => {
  for(const value of [null, {}, {...snapshot, v: 2}, {...snapshot, generator: "other"}])
    assert.throws(() => createMarbleObjectDownload(value), /versioned/);
  for(const value of [NaN, Infinity, undefined, 1n, () => {}, Symbol("x")])
    assert.throws(() => createMarbleObjectDownload({...snapshot, invalid: value}), /non-JSON/);
  const cyclic = {...snapshot};cyclic.self = cyclic;
  assert.throws(() => createMarbleObjectDownload(cyclic), /circular/i);
});

test("live PNG artifact retains the accepted Blob and exact bytes", async () => {
  const artifact = await createLivePngDownload(png, {name: snapshot.name, timestamp, digest});
  assert.equal(artifact.blob, png);
  assert.equal(artifact.blob.type, "image/png");
  assert.deepEqual(Buffer.from(await artifact.blob.arrayBuffer()), pngBytes);
  assert.equal(artifact.filename, "marble-Cafe-blue-marble-live-20261109T162345678Z-abcdef012345.png");
});

test("live PNG export refuses absent frames, non-PNG MIME and invalid PNG bytes", async () => {
  for(const blob of [null, undefined, new Blob(), {type: "image/png", size: 100}])
    await assert.rejects(createLivePngDownload(blob), /No accepted live PNG/);
  await assert.rejects(createLivePngDownload(new Blob([pngBytes], {type: "image/jpeg"})), /not a PNG/);
  for(const bytes of ["<html>error</html>", new Uint8Array([137, 80])])
    await assert.rejects(createLivePngDownload(new Blob([bytes], {type: "image/png"})), /signature/);
});

test("downloads use the exact artifact, then remove anchors and revoke URLs", async () => {
  const env = browser();
  const jsonName = await downloadMarbleObject(snapshot, env);
  const pngName = await downloadLivePng(png, {...env, name: snapshot.name});
  assert.deepEqual(env.downloads.map(download => download.filename), [jsonName, pngName]);
  assert.deepEqual(JSON.parse(await env.blobs[0].text()), snapshot);
  assert.equal(env.blobs[1], png);
  assert.deepEqual(env.events, ["url", "append", "click", "delay", "remove", "revoke",
    "url", "append", "click", "delay", "remove", "revoke"]);
  assert.equal(env.active.size, 0);
});

test("download failures reject visibly and never leak allocated URLs", async () => {
  for(const fail of ["url", "create", "append", "click", "delay"]) {
    const env = browser({fail});
    await assert.rejects(downloadMarbleObject(snapshot, env), new RegExp(`${fail} failed`));
    assert.equal(env.active.size, 0, fail);
    assert.equal(env.events.filter(event => event === "revoke").length, fail === "url" ? 0 : 1);
  }
  await assert.rejects(downloadMarbleObject(snapshot, {document: null}), /not available/);
  const env = browser();
  await assert.rejects(downloadLivePng(null, env), /No accepted live PNG/);
  assert.deepEqual(env.events, []);
  await assert.rejects(downloadMarbleObject({...snapshot, v: 0}, env), /versioned/);
  assert.deepEqual(env.events, []);
});

test("repeated exports own separate URLs and capture JSON bytes before later edits", async () => {
  const env = browser(), release = [];
  env.delay = () => new Promise(resolve => release.push(resolve));
  const editable = structuredClone(snapshot);
  const first = downloadMarbleObject(editable, env);
  editable.name = "Changed after click";
  const second = downloadMarbleObject(editable, env);
  assert.equal(env.active.size, 2);
  assert.equal(JSON.parse(await env.blobs[0].text()).name, snapshot.name);
  assert.equal(JSON.parse(await env.blobs[1].text()).name, "Changed after click");
  release[0]();await first;
  assert.equal(env.active.size, 1);
  assert.equal(env.active.has(env.downloads[1].url), true);
  release[1]();await second;
  assert.equal(env.active.size, 0);
});
