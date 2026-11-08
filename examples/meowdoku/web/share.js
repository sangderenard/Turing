// SHARE LINKS and the VIEW-ONLY LIVE STREAM.
//
// A shared link carries a whole configured view, as a small JSON document:
//   g  the game id            b  its board id (a Meowdoku puzzle)
//   c  the server-side controls   { spec id: value }   (the session's chamber, size, species, ...)
//   l  the page-local controls    { spec id: value }   (gravity scale, case angle, spin, colour, style, backdrop, panes, ...)
//   ls the page's own persisted settings (every meowdoku.* / meowdoku_* localStorage key, minus the excluded ones below)
//   cam the camera pose
// `c` and `l` are enumerated from the page's specs (app.js's pageSpecs) by ONE function, serializeSpecs, so a control added to
// the specs is carried with no change here.  `ls` is every page-owned storage key, which is what the page reads at start-up.
//
// Opening a link PREVIEWS it: the page's storage reads and writes are redirected to a memory overlay (installStorageOverlay),
// so the shared settings are in force from the first line of app.js (no flash of defaults) and the viewer's own saved settings
// are never overwritten.  "Keep these settings" copies the overlay into the real storage.  Chosen: preview-only by default.
//
// Excluded from every shared view (SECRET_KEY): any key or spec id that looks like a credential, a session, a player identity
// or a location (token, secret, password, credential, auth, cookie, session, api key, lichess, player, geo).  The Lichess token
// lives in the server's build/meowdoku/lichess_token.txt and never reaches the page; meowdoku_player (the player's name) and
// meowdoku_geo (the saved location) are the page's two storage keys that match and are never shared.
//
// Encoding: compact JSON in the URL fragment, `#s=` + (z|j) + base64url (z = deflate-raw through CompressionStream, j = plain
// when that is not available), so nothing is sent to the server.  Over MAX_FRAGMENT characters the view is stored by the
// server (POST /api/share, content addressed, size capped) and the link is `#share=<id>`.  A live view is `#watch=<token>`.
//
// Nothing here evaluates anything: the view is JSON, every value is validated through its spec's own min / max / options.

import { decompressOctets } from "./octet_codec.js";      // [LIVEVIEW] the fountain keyframe is served through the octet codec

export const MAX_FRAGMENT = 1800;
export const VIEWER_HEADER = "X-Meowdoku-Viewer";
const SECRET_KEY = /token|secret|passw|credential|auth|cookie|session|api[_-]?key|lichess|player|geo|gameConfig/i;
const SHARED_KEY = /^meowdoku[._][A-Za-z0-9_.\-]{1,80}$/;
const VALUE_KINDS = new Set(["range", "toggle", "select", "color"]);
const FORBIDDEN_PROPS = new Set(["__proto__", "constructor", "prototype"]);
const POLL_MS = 250, PUSH_MS = 250, PUSH_RETRY_MS = 5000;     // [LIVEVIEW] POLL_MS 400 -> 250: a board move shows within ~0.35 s

// ---------------------------------------------------------------- values (pure: tested in node)
export const isSharedKey = (key) => typeof key === "string" && SHARED_KEY.test(key) && !SECRET_KEY.test(key);

// the specs' value kinds a view may carry
export const sharable = (spec) => !!spec && VALUE_KINDS.has(spec.kind) && typeof spec.id === "string" && !SECRET_KEY.test(spec.id);

// A raw value, validated and clamped through the spec's own limits.  undefined = refuse (the caller keeps the current value).
export function clampValue(spec, raw) {
  switch (spec.kind) {
    case "toggle": return typeof raw === "boolean" ? raw : raw === 0 || raw === 1 ? raw === 1 : undefined;
    case "range": {
      if (typeof raw !== "number" || !Number.isFinite(raw)) return undefined;
      const lo = Number.isFinite(spec.min) ? spec.min : -Infinity, hi = Number.isFinite(spec.max) ? spec.max : Infinity;
      return Math.min(hi, Math.max(lo, raw));
    }
    case "select": {
      const options = (spec.options || []).map((o) => (Array.isArray(o) ? o[0] : o));
      if (options.length) { const hit = options.find((o) => String(o) === String(raw)); return hit; }
      // options filled in by the page later (materials, species): a short plain token or a number
      if (typeof raw === "number" && Number.isFinite(raw)) return raw;
      return typeof raw === "string" && /^[\w@.\-: ]{0,64}$/.test(raw) ? raw : undefined;
    }
    case "color": return typeof raw === "string" && /^#[0-9a-fA-F]{6}$/.test(raw) ? raw : undefined;
    default: return undefined;
  }
}

// THE ONE SERIALIZER.  `specs` is the page's spec list (pageSpecs(game.controls_ui)); a spec with `local` is the page's own and
// its value comes from `localValue`, any other from the spec itself (the server's current value).
export function serializeSpecs(specs, localValue) {
  const c = {}, l = {};
  for (const s of specs || []) {
    if (!sharable(s)) continue;
    const v = s.local ? localValue(s) : s.value;
    if (v === undefined || v === null || (typeof v === "number" && !Number.isFinite(v))) continue;
    (s.local ? l : c)[s.id] = v;
  }
  return { c, l };
}

// the page's own persisted settings
export function serializeStorage(read) {
  const ls = {};
  for (const [key, value] of read()) if (isSharedKey(key) && typeof value === "string" && value.length <= 4096) ls[key] = value;
  return ls;
}

// A stored value the page parses as JSON and Object.assign()s: only own plain primitives survive (no __proto__ / constructor).
function cleanStored(value) {
  if (typeof value !== "string" || value.length > 4096) return undefined;
  const t = value.trim();
  if (t[0] !== "{") return value;
  let parsed; try { parsed = JSON.parse(t); } catch { return undefined; }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
  const out = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (FORBIDDEN_PROPS.has(k)) continue;
    if (typeof v === "number" && Number.isFinite(v)) out[k] = Math.max(-1e6, Math.min(1e6, v));
    else if (typeof v === "boolean" || (typeof v === "string" && v.length <= 64)) out[k] = v;
  }
  return JSON.stringify(out);
}

const finiteOr = (v, d) => (typeof v === "number" && Number.isFinite(v) ? v : d);

// A decoded view -> the validated parts.  Unknown keys are ignored; every value is clamped through its spec.
export function parseView(view, specs) {
  const out = { g: null, b: null, c: {}, l: {}, ls: {}, ui: {}, cam: null, rt: null };
  if (!view || typeof view !== "object" || Array.isArray(view)) return out;
  if (typeof view.g === "string" && /^[a-z0-9_]{1,40}$/.test(view.g)) out.g = view.g;
  if (typeof view.b === "string" && /^[\w\-.:]{1,80}$/.test(view.b)) out.b = view.b;
  const byId = new Map((specs || []).filter(sharable).map((s) => [s.id, s]));
  for (const part of ["c", "l"]) {
    const src = view[part];
    if (!src || typeof src !== "object") continue;
    for (const [id, raw] of Object.entries(src)) {
      if (FORBIDDEN_PROPS.has(id)) continue;
      const spec = byId.get(id);
      if (!spec || !!spec.local !== (part === "l")) continue;
      const v = clampValue(spec, raw);
      if (v !== undefined) out[part][id] = v;
    }
  }
  if (view.ls && typeof view.ls === "object") {
    for (const [key, raw] of Object.entries(view.ls)) {
      if (!isSharedKey(key)) continue;
      const v = cleanStored(raw);
      if (v !== undefined) out.ls[key] = v;
    }
  }
  if (view.ui && typeof view.ui === "object" && !Array.isArray(view.ui)) {
    for (const [id, raw] of Object.entries(view.ui)) {
      const rule = TOP_UI[id];
      if (!rule) continue;
      if (rule === "check" && typeof raw === "boolean") out.ui[id] = raw;
      else if (rule === "number" && typeof raw === "number" && Number.isFinite(raw) && raw >= -1000 && raw <= 1000) out.ui[id] = raw;
      else if (rule === "color" && typeof raw === "string" && /^#[0-9a-fA-F]{6}$/.test(raw)) out.ui[id] = raw;
      else if (Array.isArray(rule) && typeof raw === "string" && rule.includes(raw)) out.ui[id] = raw;
      else if (rule === "clock" && typeof raw === "string" && /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(raw)) out.ui[id] = raw;
    }
  }
  const cam = view.cam;
  if (cam && typeof cam === "object") {
    const lim = { yaw: 1e3, pitch: 1.6, lz: 6, px: 1e4, pz: 1e4 };
    out.cam = {};
    for (const k of Object.keys(lim)) out.cam[k] = Math.max(-lim[k], Math.min(lim[k], finiteOr(cam[k], 0)));
    out.cam.auto = cam.auto === true;
  }
  const g = view.rt && view.rt.gravity;
  if (Array.isArray(g) && g.length >= 2 && g.every((x) => typeof x === "number" && Number.isFinite(x))) out.rt = { gravity: g.slice(0, 3).map((x) => Math.max(-4, Math.min(4, x))) };
  else if (view.rt && view.rt.gravity === null) out.rt = { gravity: null };
  return out;
}

// ---------------------------------------------------------------- encoding
const b64 = (bytes) => { let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); };
const unb64 = (text) => { const s = atob(text.replace(/-/g, "+").replace(/_/g, "/")); const out = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i); return out; };
async function pipe(bytes, stream) {
  const out = new Response(new Blob([bytes]).stream().pipeThrough(stream));
  return new Uint8Array(await out.arrayBuffer());
}
export async function encodeView(view) {
  const raw = new TextEncoder().encode(JSON.stringify(view));
  if (typeof CompressionStream !== "undefined") {
    try { return "z" + b64(await pipe(raw, new CompressionStream("deflate-raw"))); } catch { /* fall through to plain */ }
  }
  return "j" + b64(raw);
}
export async function decodeView(text) {
  if (typeof text !== "string" || text.length < 2 || text.length > 200000) throw new Error("malformed share");
  const body = unb64(text.slice(1));
  let raw;
  if (text[0] === "z") {
    if (typeof DecompressionStream === "undefined") throw new Error("this browser cannot open a compressed link");
    raw = await pipe(body, new DecompressionStream("deflate-raw"));
  } else if (text[0] === "j") raw = body;
  else throw new Error("malformed share");
  if (raw.length > 262144) throw new Error("share too large");
  return JSON.parse(new TextDecoder().decode(raw));
}

// ---------------------------------------------------------------- [LIVEVIEW] the fountain position keyframe (MFK1)
// A viewer runs the owner's fountain law locally from the owner's inputs, and drifts: its inputs land a poll later, its steps start
// later.  Every KEY_MS the owner reads its state columns (the fountain's x, y, vx, vy: one event-level readback, never per frame),
// quantizes them in the loading animation's manner (the value shifted up so the low bits are zero and the codec drops them), and
// posts them; the server packs them with the octet codec (loading_anim.pack) and hands each viewer the newest; the viewer decodes
// (octet_codec.js) and writes them into its own state columns.  At most KEY_MAX_IDS ids are sent (a stride over the ids).
//   MFK1: "MFK1" u8 ver=1 u8 0 u16 stride u32 steps u32 total u16 count, 8 x f32 ranges (x lo hi, y lo hi, vx lo hi, vy lo hi), then
//   four planar u16 columns of `count` (x, y at 12 bits, vx, vy at 10 bits, each shifted up to u16).
export const KEY_MS = 5000, KEY_MAX_IDS = 4096, KEY_HEADER = 50;
const KEY_BITS = { x: 12, y: 12, vx: 10, vy: 10 }, KEY_NAMES = ["x", "y", "vx", "vy"];
export function encodeKey({ steps, total, stride, cols }) {
  const count = Math.ceil(total / stride), range = {};
  for (const name of KEY_NAMES) {
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < count; j++) { const v = cols[name][j * stride]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (name[0] === "v") { hi = Math.max(Math.abs(lo), Math.abs(hi), 1e-6); lo = -hi; }
    if (!(hi > lo)) hi = lo + 1e-6;
    range[name] = [Math.fround(lo), Math.fround(hi)];
  }
  const out = new Uint8Array(KEY_HEADER + 8 * count), head = new DataView(out.buffer);
  "MFK1".split("").forEach((c, i) => head.setUint8(i, c.charCodeAt(0)));
  head.setUint8(4, 1); head.setUint16(6, stride, true); head.setUint32(8, steps >>> 0, true); head.setUint32(12, total, true); head.setUint16(16, count, true);
  KEY_NAMES.forEach((name, k) => { head.setFloat32(18 + 8 * k, range[name][0], true); head.setFloat32(22 + 8 * k, range[name][1], true); });
  KEY_NAMES.forEach((name, k) => {
    const [lo, hi] = range[name], top = (1 << KEY_BITS[name]) - 1, shift = 16 - KEY_BITS[name], at = KEY_HEADER + 2 * count * k;
    for (let j = 0; j < count; j++) {
      const u = (Math.max(0, Math.min(top, Math.round(((cols[name][j * stride] - lo) / (hi - lo)) * top))) << shift) & 0xffff;
      out[at + j] = u & 255; out[at + count + j] = u >> 8;
    }
  });
  return out;
}
export function decodeKey(bytes) {
  if (bytes.length < KEY_HEADER || String.fromCharCode(...bytes.subarray(0, 4)) !== "MFK1") throw new Error("not a fountain keyframe");
  const head = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const stride = head.getUint16(6, true), steps = head.getUint32(8, true), total = head.getUint32(12, true), count = head.getUint16(16, true);
  if (!stride || !count || bytes.length !== KEY_HEADER + 8 * count || count !== Math.ceil(total / stride)) throw new Error("malformed fountain keyframe");
  const out = { stride, steps, total, count };
  KEY_NAMES.forEach((name, k) => {
    const lo = head.getFloat32(18 + 8 * k, true), hi = head.getFloat32(22 + 8 * k, true), top = (1 << KEY_BITS[name]) - 1, shift = 16 - KEY_BITS[name], at = KEY_HEADER + 2 * count * k;
    const col = new Float32Array(count);
    for (let j = 0; j < count; j++) col[j] = lo + (((bytes[at + j] | (bytes[at + count + j] << 8)) >> shift) / top) * (hi - lo);
    out[name] = col;
  });
  return out;
}
const b64std = (bytes) => { let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const unb64std = (text) => { const s = atob(text), out = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i); return out; };
// a viewer takes the keyframe: the sampled ids' columns replaced, the rest left to the viewer's own law
async function applyKey(host, packet) {
  const f = host.fountain; if (!f || !f.ready()) return false;
  const k = decodeKey(decompressOctets(packet));
  for (const name of KEY_NAMES) {
    let col;
    if (k.stride === 1) col = k[name];
    else { col = (await f.read([name]))[name]; if (!col || col.length !== k.total) return false; for (let j = 0; j < k.count; j++) col[j * k.stride] = k[name][j]; }
    if (col.length !== k.total || !f.write(name, col)) return false;
  }
  return true;
}

// ---------------------------------------------------------------- the page-side module state
const inBrowser = typeof window !== "undefined" && typeof document !== "undefined" && typeof location !== "undefined";
const TOP_UI = {
  "sky-mode": ["auto", "day", "night"], "sky-place": ["here", "honolulu", "los_angeles", "chicago", "new_york", "reykjavik", "london", "cairo", "mumbai", "beijing", "tokyo", "sydney", "ushuaia"],
  "sky-clock": "clock", "auto-lamp": "check", "lamp-color": ["warm", "amber", "neutral", "cool"],
  wobble: "check", "wobble-k": "number", "wobble-gpu": "check", tilt: "number", curve: "number", saturation: "number", round: "number", crease: "number",
  arcade: "check", plate: "check", projection: "check", autorestart: "check", pads: "check",
  "marble-preview": "check", "marble-eye": ["blue", "green", "amber"], "marble-width": "number", "marble-curve": "number", "marble-facets": ["12", "24", "40"],
  "marble-phone-tilt": "check", "marble-camera-tilt": "check", "marble-follow": "check", "marble-watch-case": "check",
  "marble-backlit": "check", "marble-spot": "check", "marble-light-mode": ["lit", "glow", "both"],
  "marble-glow-color": "color", "marble-glow-strength": "number", light: ["sun", "pointer"], flat: "check",
};
function captureTopUI() {
  const ui = {};
  for (const [id, rule] of Object.entries(TOP_UI)) {
    const node = document.getElementById(id); if (!node) continue;
    ui[id] = id === "light" ? (node.textContent.includes("pointer") ? "pointer" : "sun")
      : id === "flat" ? node.textContent.includes("on")
      : rule === "check" ? node.checked : rule === "number" ? Number(node.value) : node.value;
  }
  return ui;
}
function applyTopUI(ui) {
  for (const [id, value] of Object.entries(ui)) {
    const node = document.getElementById(id); if (!node) continue;
    if (id === "light") { if (node.textContent.includes("pointer") !== (value === "pointer")) node.click(); continue; }
    if (id === "flat") { if (node.textContent.includes("on") !== value) node.click(); continue; }
    if (TOP_UI[id] === "check") { if (node.checked === value) continue; node.checked = value; }
    else { if (node.value === String(value)) continue; node.value = String(value); }
    node.dispatchEvent(new Event("input", { bubbles: true }));
    node.dispatchEvent(new Event("change", { bubbles: true }));
  }
}
const TOP_UI_DEFAULTS = inBrowser ? captureTopUI() : {};
export const share = { mode: "owner", view: null, parsed: null, token: null, viewOnly: false, host: null, overlay: null, error: null,
  socketOpen: false, ended: false, rxAt: 0, rxBytes: 0 };      // [LIVEVIEW] the viewer's socket state

// Redirect the page's storage reads / writes of its own keys to memory (preview and viewer modes).  Keys the view does not carry
// read as unset (the defaults), so the view is shown as the owner has it, not mixed with the viewer's own settings.
function installStorageOverlay(initial) {
  const proto = Storage.prototype, real = { get: proto.getItem, set: proto.setItem, remove: proto.removeItem };
  const mem = new Map(Object.entries(initial));
  const own = (key) => SHARED_KEY.test(String(key)) && !SECRET_KEY.test(String(key));
  proto.getItem = function (key) { return this === localStorage && own(key) ? (mem.has(String(key)) ? mem.get(String(key)) : null) : real.get.call(this, key); };
  proto.setItem = function (key, value) { if (this === localStorage && own(key)) mem.set(String(key), String(value)); else real.set.call(this, key, value); };
  proto.removeItem = function (key) { if (this === localStorage && own(key)) mem.delete(String(key)); else real.remove.call(this, key); };
  share.overlay = {
    mem,
    keep() { for (const [k, v] of mem) real.set.call(localStorage, k, v); },
    realEntries() { const out = []; try { for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); out.push([k, real.get.call(localStorage, k)]); } } catch { /* blocked */ } return out; },
  };
}
// the current page settings (the overlay when previewing)
function storageEntries() {
  if (share.overlay) return [...share.overlay.mem.entries()];
  const out = []; try { for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); out.push([k, localStorage.getItem(k)]); } } catch { /* blocked */ }
  return out;
}

// In viewer mode nothing is sent but the read: same-origin /api/ calls carry the viewer header (the server refuses all but the
// catalog and /api/spectate/watch for it), and a WebSocket cannot be opened.
function installViewerGuards(token) {
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (/^(https?:\/\/[^/]+)?\/api\//.test(url)) {
      if (!/\/api\/(catalog|spectate\/watch)/.test(url)) return Promise.reject(new Error("view-only: this link cannot send inputs or controls"));
      return realFetch(input, { ...init, headers: { ...(init.headers || {}), [VIEWER_HEADER]: token } });
    }
    return realFetch(input, init);
  };
  // [LIVEVIEW] the only socket a viewer may open is the read-only spectate socket.  app.js asks for its own "/ws" (the owner's
  // input socket); that request is answered by /ws/spectate with the token in the subprotocol (a browser cannot set a header,
  // and the token stays out of the URL).  The instance's send() is dropped here AND refused by the server (1008).
  const RealWS = window.WebSocket;
  window.WebSocket = function (url) {
    const u = new URL(url, location.href);
    if (u.pathname !== "/ws" && u.pathname !== "/ws/spectate") throw new Error("view-only: no input socket");
    const ws = new RealWS(`${u.protocol}//${u.host}/ws/spectate`, ["mdv." + token]);
    ws.send = () => {};
    ws.addEventListener("open", () => { share.socketOpen = true; share.rxAt = performance.now(); });
    ws.addEventListener("message", (e) => { share.rxAt = performance.now(); share.rxBytes += typeof e.data === "string" ? e.data.length : e.data.byteLength || 0; });
    ws.addEventListener("close", (e) => { share.socketOpen = false; if (e.code === 4410 || e.code === 1008) share.ended = true; });
    return ws;
  };
}

function hashParams() { return new URLSearchParams(location.hash.replace(/^#/, "")); }

async function loadIncoming() {
  const h = hashParams();
  if (h.get("watch")) {
    share.mode = "viewer"; share.token = h.get("watch"); share.viewOnly = true;
    installViewerGuards(share.token);
    try {                       // the owner's persisted settings must be in force before app.js reads its storage
      const r = await fetch("/api/spectate/watch?rev=", { headers: { [VIEWER_HEADER]: share.token } });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.status);
      share.first = d;
    } catch (e) { share.error = String(e.message || e); share.first = null; }
    return;
  }
  try {
    let view = null;
    if (h.get("p") || h.get("s")) view = await decodeView(h.get("p") || h.get("s"));
    else if (h.get("play") || h.get("share")) {
      const r = await fetch(`/api/share/${encodeURIComponent(h.get("play") || h.get("share"))}`);
      if (!r.ok) throw new Error((await r.json()).error || r.status);
      view = await r.json();
    }
    if (view) { share.mode = h.get("p") || h.get("play") ? "play" : "preview"; share.view = view; if (share.mode === "play") document.body.classList.add("play-link"); }
  } catch (e) { share.error = String(e.message || e); console.warn("[meowdoku] share link:", share.error); }
}

// ---------------------------------------------------------------- the UI and the host (app.js) hooks
const el = (tag, props = {}, ...kids) => { const n = Object.assign(document.createElement(tag), props); n.append(...kids); return n; };
const STYLE = "position:fixed;z-index:60;right:8px;bottom:8px;max-width:min(28rem,94vw);background:var(--ctl,#223);color:var(--fg,#eee);border:1px solid var(--line,#556);border-radius:8px;padding:8px 10px;font:13px/1.35 system-ui,sans-serif;display:grid;gap:6px";

function panel() {
  let p = document.getElementById("share-panel");
  if (!p) { p = el("div", { id: "share-panel" }); p.style.cssText = STYLE; document.body.append(p); }
  return p;
}
function showPanel(...nodes) { const p = panel(); p.replaceChildren(...nodes, el("button", { textContent: "close", onclick: () => p.remove() })); }

async function copyText(text, box) {
  box.select();
  try { await navigator.clipboard.writeText(text); return true; } catch { /* insecure context / iOS Safari: fall back */ }
  try { box.setSelectionRange(0, text.length); return document.execCommand("copy"); } catch { return false; }
}

function currentView(gameKey = null) {
  const h = share.host, game = h.getGame();
  const { c, l } = serializeSpecs(h.pageSpecs(game && game.controls_ui), h.localValue);
  const view = { v: 1, g: gameKey || (game?.tipjar ? "tipjar" : game?.game || null), c, l, ls: serializeStorage(storageEntries), ui: captureTopUI(), cam: h.camera.get(), rt: h.runtime.get() };
  if (game && game.game === "meowdoku" && typeof game.id === "string") view.b = game.id;
  return view;
}
export function buildView(gameKey = null) { return currentView(gameKey); }

async function makeLink(kind = "view") {
  const view = currentView(), base = location.origin + location.pathname;
  const enc = await encodeView(view);
  if (enc.length + 3 <= MAX_FRAGMENT) return { url: `${base}#${kind === "play" ? "p" : "s"}=${enc}`, how: `in the link (${enc.length} characters; nothing is sent to the server)` };
  try {
    const r = await fetch("/api/share", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ view }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.status);
    return { url: `${base}#${kind === "play" ? "play" : "share"}=${data.id}`, how: `stored on the server as ${data.id} (the view was ${enc.length} characters)` };
  } catch (e) {
    return { url: `${base}#${kind === "play" ? "p" : "s"}=${enc}`, how: `the server would not store it (${e.message}); the link is long (${enc.length} characters)` };
  }
}

async function onShare() {
  if (!share.host.getGame()) return;
  const { url, how } = await makeLink();
  const box = el("textarea", { value: url, readOnly: true, rows: 3 }); box.style.cssText = "width:100%;font:12px monospace";
  const note = el("div", { textContent: "" });
  const copy = el("button", { textContent: "Copy link", onclick: async () => { note.textContent = (await copyText(url, box)) ? "copied" : "press Ctrl/Cmd-C to copy the selected text"; } });
  showPanel(el("b", { textContent: "Share this view" }), box, el("div", { textContent: `Holds ${how}. Opening it shows this game, its controls, the page settings and the camera; the viewer's own saved settings are left alone.` }), copy, note);
  box.focus(); box.select();
}
async function onShareGame() {
  if (!share.host.getGame()) return;
  const { url, how } = await makeLink("play");
  const box = el("textarea", { value: url, readOnly: true, rows: 3 }); box.style.cssText = "width:100%;font:12px monospace";
  const note = el("div", { textContent: "" });
  const copy = el("button", { textContent: "Copy link", onclick: async () => { note.textContent = (await copyText(url, box)) ? "copied" : "press Ctrl/Cmd-C to copy the selected text"; } });
  showPanel(el("b", { textContent: "Share game" }), box, el("div", { textContent: `Holds ${how}. Opens an independent game with this setup and a compact play interface; the visitor keeps their own player identity.` }), copy, note);
  box.focus(); box.select();
}

// ---- the owner's live share
const live = { on: false, token: null, timer: null, statusTimer: null, last: "", busy: false, tried: "", retryAt: 0 };
const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(async (r) => { const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status); return d; });
function liveView() {
  const h = share.host, game = h.getGame();
  const { l } = serializeSpecs(h.pageSpecs(game && game.controls_ui), h.localValue);
  return { v: 1, l, rt: h.runtime.get(), ls: serializeStorage(storageEntries) };
}
function startPushing() {
  if (live.timer) return;
  live.keyTimer = setInterval(pushKey, KEY_MS);
  live.timer = setInterval(async () => {
    if (live.busy || !share.host.getGame()) return;
    const text = JSON.stringify(liveView());
    // [LIVEVIEW] push on change only, and never retry a refused payload in a loop: a payload the server refused (e.g. 413 too
    // large) used to be re-sent every PUSH_MS forever (dozens of POSTs in a row).  It is remembered as attempted and not
    // re-sent until it changes or PUSH_RETRY_MS has passed.
    if (text === live.last || (text === live.tried && performance.now() < live.retryAt)) return;
    live.busy = true; live.tried = text;
    try { await post("/api/spectate/live", { live: JSON.parse(text) }); live.last = text; }
    catch (e) { live.retryAt = performance.now() + PUSH_RETRY_MS; if (/not sharing/.test(String(e.message))) stopPushing(); }
    finally { live.busy = false; }
  }, PUSH_MS);
}
function stopPushing() { clearInterval(live.timer); clearInterval(live.statusTimer); clearInterval(live.keyTimer); live.timer = live.statusTimer = live.keyTimer = null; live.on = false; live.last = ""; }
// [LIVEVIEW] the owner's fountain position keyframe, every KEY_MS while a viewer exists (the server says how many watch)
async function pushKey() {
  const f = share.host && share.host.fountain;
  if (!live.on || !f || !f.ready() || live.keyBusy) return;
  live.keyBusy = true;
  try {
    const s = await post("/api/spectate", { op: "status" });                       // (no viewer: no readback)
    if (!s.sharing) { stopPushing(); return; }
    if (!s.viewers) return;
    const steps = f.steps(), cols = await f.read(KEY_NAMES), total = cols.x.length, stride = Math.max(1, Math.ceil(total / KEY_MAX_IDS));
    const raw = encodeKey({ steps, total, stride, cols });
    await post("/api/spectate/key", { key: b64std(raw) });
    live.keys = (live.keys || 0) + 1; live.keyRaw = raw.length;
  } catch (e) { if (/not sharing/.test(String(e.message))) stopPushing(); } finally { live.keyBusy = false; }
}
function renderLive(status, box) {
  const url = `${location.origin}${location.pathname}#watch=${status.token}`;
  const link = el("textarea", { value: url, readOnly: true, rows: 2 }); link.style.cssText = "width:100%;font:12px monospace";
  const note = el("div", { textContent: "" });
  const stats = el("div", { textContent: statsText(status) });
  box.replaceChildren(el("b", { textContent: "Live view (read-only)" }), link,
    el("div", { textContent: "Viewers see this session's game and controls and run the same law locally with your inputs. They cannot send inputs or controls; the server refuses them." }),
    el("button", { textContent: "Copy link", onclick: async () => { note.textContent = (await copyText(url, link)) ? "copied" : "press Ctrl/Cmd-C to copy the selected text"; } }),
    stats, note,
    el("button", { textContent: "Stop sharing", onclick: async () => { await post("/api/spectate", { op: "stop" }).catch(() => {}); stopPushing(); box.remove(); } }),
    el("button", { textContent: "close", onclick: () => box.remove() }));
  return stats;
}
function statsText(s) {
  const per = (s.per_viewer || []).map((v) => v.bytes_per_s);
  const mean = per.length ? per.reduce((a, b) => a + b, 0) / per.length : 0;
  return `${s.viewers} of ${s.cap} viewers; about ${(mean / 1024).toFixed(2)} kB/s per viewer`;
}
async function onShareLive() {
  if (!share.host.getGame()) return;
  try {
    const status = await post("/api/spectate", { op: "start" });
    live.on = true; live.token = status.token;
    startPushing();
    const box = panel(); const stats = renderLive(status, box);
    clearInterval(live.statusTimer);
    live.statusTimer = setInterval(async () => { try { const s = await post("/api/spectate", { op: "status" }); if (!s.sharing) { stopPushing(); return; } stats.textContent = statsText(s); } catch { /* retry */ } }, 3000);
  } catch (e) { showPanel(el("div", { textContent: `Could not start sharing: ${e.message}` })); }
}
async function resumeLive() {   // a reload of the owner's page keeps the server-side share running
  try { const s = await post("/api/spectate", { op: "status" }); if (s.sharing) { live.on = true; live.token = s.token; startPushing(); } } catch { /* no server */ }
}

function addButtons() {
  const anchor = document.getElementById("new");
  const mk = (text, title, fn) => el("button", { textContent: text, title, onclick: fn });
  const a = mk("Share", "a link that restores this whole view", onShare), p = mk("Share game", "an independent game with this configuration and compact controls", onShareGame), b = mk("Share live", "a read-only link that follows this session live", onShareLive);
  a.id = "share-view-button"; b.id = "share-live-button";
  if (anchor) anchor.after(a, p, b); else document.body.append(a, p, b);
}
function banner(text, ...buttons) {
  document.getElementById("share-banner")?.remove();
  const n = el("div", { id: "share-banner" }, el("span", { textContent: text + " " }), ...buttons);
  n.style.cssText = "position:fixed;z-index:60;left:8px;bottom:8px;background:var(--ctl,#223);color:var(--fg,#eee);border:1px solid var(--line,#556);border-radius:8px;padding:6px 10px;font:13px system-ui,sans-serif;max-width:min(30rem,94vw)";
  document.body.append(n);
  return n;
}

// ---- applying a parsed view
function applyLocals(local) {
  const h = share.host, game = h.getGame(); if (!game) return;
  // [LIVEVIEW] ROOT CAUSE of "one picture": pageSpecs(undefined) is undefined for a game with no declared controls (every board
  // game and real-time entity game); this .filter threw inside the viewer's FIRST step, applyViewer rejected before its poll
  // loop was ever started, and bootGame fell through to the page's own "new game".  Tolerate it, and never let a step end the loop.
  const byId = new Map((h.pageSpecs(game.controls_ui) || []).filter((s) => s.local).map((s) => [s.id, s]));
  for (const [id, v] of Object.entries(local)) { const s = byId.get(id); if (s) h.localControl(s, v); }
}

async function applyConfiguration(view, st) {
  const h = share.host;
  for (const [id, raw] of Object.entries(view.c && typeof view.c === "object" ? view.c : {})) {
    const s = (h.pageSpecs(st.controls_ui) || []).find((x) => x.id === id && !x.local && sharable(x));
    const want = s && s.action ? clampValue(s, raw) : undefined;
    if (want === undefined || String(s.value) === String(want)) continue;
    st = await h.api("/api/action", h.serverPayload(s, want));
  }
  h.applyState(st);
  const full = parseView(view, h.pageSpecs(st.controls_ui));
  applyLocals(full.l);
  applyTopUI(full.ui);
  if (full.cam) h.camera.set(full.cam);
  if (full.rt && full.rt.gravity !== undefined) h.runtime.set(full.rt);
  return st;
}
const profileKey = (game) => `meowdoku.gameConfig.${game}`;
let configuredOnce = false;
export function rememberGame(game) {
  if (!inBrowser || share.mode !== "owner" || !/^[a-z0-9_]{1,40}$/.test(game) || !share.host?.getGame()) return;
  try { localStorage.setItem(profileKey(game), JSON.stringify(currentView(game))); } catch { /* private storage */ }
}
export async function configureGame(game, st) {
  if ((share.mode === "play" || share.mode === "preview") && share.view?.g === game)
    return applyConfiguration(share.view, st);
  if (share.mode !== "owner") { share.host.applyState(st); return st; }
  let view = null;
  try { view = JSON.parse(localStorage.getItem(profileKey(game)) || "null"); } catch { /* private storage */ }
  if (!view || view.g !== game) { share.host.applyState(st); if (configuredOnce) applyTopUI(TOP_UI_DEFAULTS); configuredOnce = true; return st; }
  configuredOnce = true;
  return applyConfiguration(view, st);
}
async function applyPreview() {
  const h = share.host, p = parseView(share.view, []), catalog = h.catalog();
  if (!p.g || !catalog.games.includes(p.g)) return false;
  h.selectGame(p.g);
  const payload = { game: p.g, aspect: h.aspect() };
  if (p.g === "meowdoku") { if (!p.b || !catalog.meowdoku.boards.some((x) => x.id === p.b)) return false; payload.id = p.b; }
  const st = await h.api("/api/new", payload);
  await applyConfiguration(share.view, st);
  return true;
}

// ---- the viewer
async function applyViewer() {
  const h = share.host, token = share.token;
  let vid = share.first.vid, rev = null, liveRev = -1, applied = {}, stopped = false, lastLive = {};
  // [LIVEVIEW] the badge: LIVE · watching, with the age of the last update (a polled state, or a snapshot on the socket).
  const bar = banner("");
  bar.style.borderColor = "#3c9";
  const view = { viewers: 1, rate: "", polled: 0, pollBytes: 0, since: performance.now(), err: null, lastAt: performance.now(), okAt: performance.now() };
  const badge = () => {
    if (stopped) return;
    const now = performance.now(), last = Math.max(view.lastAt, share.rxAt), age = (now - last) / 1000;
    const via = share.socketOpen ? "socket" : "poll";
    const kbs = ((view.pollBytes + share.rxBytes) / Math.max(1, (now - view.since) / 1000) / 1024).toFixed(2);
    const stale = (now - Math.max(view.okAt, share.rxAt)) / 1000 > 4 || view.err;       // (live = the link answers; an idle game is still live)
    bar.firstChild.textContent = `${stale ? "RECONNECTING" : "LIVE · watching"} · ${view.err ? view.err + " · " : ""}last update ${age.toFixed(1)} s ago · ${view.viewers} watching · ${kbs} kB/s via ${via} · read-only `;
    bar.style.borderColor = stale ? "#e93" : "#3c9";
  };
  let keySeq = 0;
  share.viewerView = view;                               // (test / debug hook: the badge's numbers)
  const heldKey = () => { const g = h.getGame(); return g ? `${g.game ?? ""}|${g.id ?? ""}` : ""; };
  const poll = async () => {
    const r = await fetch(`/api/spectate/watch?${vid ? `vid=${encodeURIComponent(vid)}&` : ""}rev=${rev || ""}&lrev=${liveRev}${share.socketOpen && h.getGame() && h.getGame().input === "realtime" ? `&nostate=1&gk=${encodeURIComponent(heldKey())}` : ""}&kseq=${keySeq}`, { headers: { [VIEWER_HEADER]: token } });
    const text = await r.text(); view.pollBytes += text.length + 180;
    const d = JSON.parse(text);
    if (!r.ok) { const e = new Error(d.error || r.status); e.status = r.status; throw e; }
    return d;
  };
  const step = (d) => {
    vid = d.vid; view.viewers = d.viewers || view.viewers; view.err = null; view.okAt = performance.now();
    if (d.state) { view.lastAt = performance.now(); rev = d.state_rev; h.applyState(d.state); h.selectGame(d.state.game); }
    if (d.live) lastLive = d.live;
    if (d.live_rev !== liveRev || d.state) {
      liveRev = d.live_rev;
      const game = h.getGame();
      const parsed = parseView({ l: lastLive.l, rt: lastLive.rt, g: game && game.game }, h.pageSpecs(game && game.controls_ui));
      // the viewer's own gravity sources are off: the owner's gravity vector is the input
      parsed.l.tilt = false; parsed.l.camgrav = false;
      const diff = {}; for (const [k, v] of Object.entries(parsed.l)) if (d.state || applied[k] !== v) diff[k] = v;
      Object.assign(applied, diff); applyLocals(diff);
      if (parsed.rt) h.runtime.set(parsed.rt);
    }
    if (d.live) view.lastAt = performance.now();
    if (d.key && d.key_seq !== keySeq) {                // the owner's newest fountain position keyframe
      keySeq = d.key_seq; view.keys = (view.keys || 0) + 1;
      applyKey(h, unb64std(d.key)).then((ok) => { if (ok) { view.lastAt = performance.now(); view.keyApplied = (view.keyApplied || 0) + 1; } }).catch((e) => console.warn("[meowdoku] live view keyframe:", e));
    }
    badge();
  };
  try { step(share.first); } catch (e) { console.warn("[meowdoku] live view first step:", e); }
  setInterval(badge, 200);
  const loop = async () => {
    while (!stopped) {
      await new Promise((ok) => setTimeout(ok, share.socketOpen && h.getGame() && h.getGame().input === "realtime" ? 1000 : POLL_MS));      // (the socket carries a real-time game; the poll carries the controls)
      try { step(await poll()); }
      catch (e) {
        if (!e.status) console.warn("[meowdoku] live view step:", e);
        if (e.status === 410 || e.status === 403 || e.status === 429) { stopped = true; bar.firstChild.textContent = `LIVE view ended: ${e.message} `; bar.style.borderColor = "#e55"; return; }
        view.err = "reconnecting"; badge();                                                // a dropped poll retries; never gives up on a network error
        await new Promise((ok) => setTimeout(ok, 1500));
      }
    }
  };
  loop();
  return true;
}

// ---------------------------------------------------------------- the entry points app.js uses
export function attach(host) {
  share.host = host;
  if (!inBrowser) return;
  if (share.mode !== "viewer") addButtons();
  if (share.mode === "owner") resumeLive();
}
// Called at boot in place of the page's default "new game".  Resolves true when a shared view was opened.
export async function bootGame() {
  if (!share.host) return false;
  try {
    if (share.mode === "viewer") return share.first ? await applyViewer() : (document.getElementById("status").textContent = share.error || "this live view is not available", true);
    if (share.mode === "preview" || share.mode === "play") {
      const ok = await applyPreview();
      if (ok) banner(share.mode === "play" ? "Playing a shared setup in your own session." : "Viewing a shared setup.",
        el("button", { textContent: "Keep these settings", onclick: () => { share.overlay.keep(); document.getElementById("share-banner")?.remove(); history.replaceState(null, "", location.pathname); } }),
        el("button", { textContent: "Close", onclick: () => document.getElementById("share-banner")?.remove() }));
      return ok;
    }
  } catch (e) { console.warn("[meowdoku] share:", e); share.error = String(e.message || e); }
  return false;
}

// Module start (before app.js runs its first line): decode the incoming link, and put its settings in force.
if (inBrowser) {
  await loadIncoming();
  if (share.mode === "preview" || share.mode === "play") {
    const early = parseView(share.view, []);           // ls needs no specs
    installStorageOverlay(early.ls);
  } else if (share.mode === "viewer") {
    installStorageOverlay(share.first ? parseView({ ls: share.first.live.ls }, []).ls : {});
  }
}
