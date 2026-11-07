// [STUDIO] The loading studio page (studio.html).
//  1. the loading screen: the real loading panel (loading_view.html -> loading_panel.js) held on a phase, playing any library entry;
//  2. re-recording: loading_params.js controls -> the game page in an iframe running loading_record.js with those parameters;
//  3. the library (/api/loading_library): thumbnails, play, compare two, and -- with the owner's token -- make active / delete.
import { DEFAULTS, normalize, toQuery, CHAMBERS, SIZES, SIZE_IDS, BITS, COLOUR_SOURCES, RAW_CAP } from "./loading_params.js";
import { decodePacket, seam, angleOf } from "./loading_anim.js";
const PHASES = { compiling: 1, downloading: 1, initializing: 1 };      // the panel's phases (loading_panel.js PHASES; importing that module would start a panel here)

const $ = (id) => document.getElementById(id);
const S = { phase: "downloading", entries: [], active: null, playing: null, A: null, B: null, params: { ...DEFAULTS }, recFrame: null, thumbs: new Map(), hover: null, deployedSizes: null };
const TOKEN_KEY = "meowdoku.studio.token";

// ------------------------------------------------------------------ the owner token (sessionStorage only; sent in a header, never a URL)
const store = { get() { try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } }, set(v) { try { v ? sessionStorage.setItem(TOKEN_KEY, v) : sessionStorage.removeItem(TOKEN_KEY); } catch { /* blocked */ } } };
const secure = location.protocol === "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
$("token").value = store.get();
$("token").addEventListener("input", () => store.set($("token").value));
$("tokenNote").textContent = secure ? "kept for this tab only" : "owner actions need HTTPS: open this page over https";

async function owner(path, digest) {
  if (!secure) throw new Error("owner actions need HTTPS");
  const token = $("token").value;                    // local (LAN) visitors need none; remote owners type it (top right)
  const headers = { "Content-Type": "application/json" }; if (token) headers["X-Meowdoku-Admin"] = token;
  const r = await fetch(path, { method: "POST", headers, body: JSON.stringify({ digest }) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

// ------------------------------------------------------------------ 1. the loading screen
const viewUrl = (phase, digest) => `loading_view.html?loading_demo=${phase}${digest ? `&loading_anim=${digest}` : ""}`;
function post(frame, msg) { try { frame.contentWindow.postMessage({ type: "loading-view", ...msg }, location.origin); } catch { /* not loaded yet */ } }
function showScreen() {
  const frame = $("screen");
  if (!frame.dataset.loaded) { frame.src = viewUrl(S.phase, S.playing); frame.dataset.loaded = "1"; }
  else post(frame, { phase: S.phase, anim: S.playing || "" });
  const e = S.entries.find((x) => x.digest === S.playing);
  $("playingNote").textContent = S.playing ? `playing ${S.playing.slice(0, 8)}${e && e.active ? " (active)" : ""}` : (S.active ? "playing the active one" : "no recording yet: the procedural hourglass");
}
function buildPhases() {
  const box = $("phases");
  for (const name of Object.keys(PHASES)) {
    const b = document.createElement("button"); b.textContent = name; b.dataset.phase = name;
    b.addEventListener("click", () => { S.phase = name; paintPhases(); showScreen(); for (const id of ["viewA", "viewB"]) post($(id), { phase: name }); });
    box.appendChild(b);
  }
  paintPhases();
}
const paintPhases = () => document.querySelectorAll("#phases button").forEach((b) => b.classList.toggle("on", b.dataset.phase === S.phase));
$("playActive").addEventListener("click", () => { S.playing = null; paintCards(); showScreen(); });

// ------------------------------------------------------------------ 2. recording controls
const CONTROLS = [
  ["Scene"],
  { k: "chamber", label: "chamber", type: "select", options: CHAMBERS },
  { k: "size", label: "sim size (ids)", type: "select", options: SIZES.map((s) => [s, `${s} (${SIZE_IDS[s].toLocaleString()})`]) },
  { k: "mix", label: "species mix (poly mix)", type: "range", min: 0, max: 1, step: 0.05, nullable: true },
  { k: "grainScale", label: "grain size (x radius)", type: "range", min: 0.5, max: 2, step: 0.05 },
  { k: "colour", label: "colour source (preview, metadata)", type: "select", options: COLOUR_SOURCES },
  ["Animation"],
  { k: "grains", label: "grains kept", type: "range", min: 16, max: 1024, step: 16 },
  { k: "fps", label: "record fps", type: "range", min: 2, max: 30, step: 1 },
  { k: "bits", label: "bit depth", type: "select", options: BITS.map((b) => [b, `${b} bit`]) },
  { k: "cap", label: "duration cap (sim s)", type: "range", min: 10, max: 900, step: 10 },
  ["Flip rule"],
  { k: "settle", label: "settle threshold (x law KE/id)", type: "range", min: 1, max: 400, step: 1 },
  { k: "turns", label: "turns (even)", type: "range", min: 2, max: 12, step: 2 },
  { k: "flipSeconds", label: "flip speed (turn takes s)", type: "range", min: 0.1, max: 2, step: 0.05 },
  { k: "flipAxis", label: "flip axis", type: "select", options: [["y", "world Y / case normal (2-D)"], ["x", "world X / horizontal (3-D)"], ["z", "world Z / vertical (3-D)"]] },
  { k: "flipDirection", label: "flip direction", type: "select", options: [[1, "one way"], [-1, "reverse"]] },
  { k: "speed", label: "sim speed", type: "select", options: [[0, "max"], [1, "1x"], [2, "2x"], [4, "4x"]] },
  ["Budget"],
  { k: "fit", label: "tighten automatically to fit ~900 KB", type: "check" },
];
function buildControls() {
  const box = $("controls"); box.textContent = "";
  for (const c of CONTROLS) {
    if (Array.isArray(c)) { const h = document.createElement("h3"); h.textContent = c[0]; box.appendChild(h); continue; }
    const lab = document.createElement("label"), head = document.createElement("span"), out = document.createElement("b");
    head.append(c.label, out); lab.appendChild(head);
    let input;
    if (c.type === "select") {
      input = document.createElement("select");
      for (const [v, t] of c.options) { const o = document.createElement("option"); o.value = v; o.textContent = t; input.appendChild(o); }
    } else if (c.type === "check") {
      input = document.createElement("input"); input.type = "checkbox";
    } else {
      input = document.createElement("input"); input.type = "range"; input.min = c.min; input.max = c.max; input.step = c.step;
    }
    c.input = input; c.out = out; lab.appendChild(input);
    if (c.nullable) {                       // "default" = the game's own mix
      const d = document.createElement("label"); d.style.gridAutoFlow = "column"; d.style.justifyContent = "start"; d.style.alignItems = "center"; d.style.gap = "6px";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.id = "mixDefault"; d.append(cb, "game default"); lab.appendChild(d);
      cb.addEventListener("change", () => { S.params.mix = cb.checked ? null : Number(input.value); paintControls(); });
    }
    input.addEventListener("input", () => {
      const v = c.type === "check" ? input.checked : (c.type === "select" ? (isNaN(Number(input.value)) ? input.value : Number(input.value)) : Number(input.value));
      S.params[c.k] = v; if (c.k === "mix") $("mixDefault").checked = false;
      S.params = { ...normalize(S.params), fit: S.params.fit }; paintControls();
    });
    box.appendChild(lab);
  }
  paintControls();
}
function paintControls() {
  for (const c of CONTROLS) {
    if (Array.isArray(c)) continue;
    const v = S.params[c.k];
    if (c.type === "check") c.input.checked = !!v;
    else if (c.k === "mix") { c.input.value = v === null ? 0.5 : v; $("mixDefault").checked = v === null; c.input.disabled = v === null; }
    else c.input.value = String(v);
    if (c.k === "size" && S.deployedSizes) for (const o of c.input.options) o.disabled = !S.deployedSizes[S.params.flipAxis].has(o.value);
    c.out.textContent = c.type === "select" || c.type === "check" ? "" : (c.k === "mix" && v === null ? "default" : Number(v).toFixed(c.step < 1 ? 2 : 0));
  }
}
async function loadSizes() {
  try {
    const r = await fetch("fountain.manifest.json", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const manifest = await r.json();
    let three = new Set();
    try {
      const r3 = await fetch("fountain3d.manifest.json", { cache: "no-store" });
      if (r3.ok) {
        const m3 = await r3.json(), inputs = new Set(m3.law && m3.law.runtime_inputs || []);
        if (["case_omega_x", "case_omega_y", "case_domega_x", "case_domega_y"].every((name) => inputs.has(name))) three = new Set(Object.keys(m3.sizes || {}));
      }
    } catch { /* the 3-D product has not been built yet */ }
    S.deployedSizes = { y: new Set(Object.keys(manifest.sizes || {})), x: three, z: three };
    paintControls();
  } catch (error) { recMsg(`fountain products unavailable: ${error.message}`, "err"); }
}
$("reset").addEventListener("click", () => { S.params = { ...DEFAULTS }; paintControls(); });

const kb = (n) => (n < 10240 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
function stats(rows) { const dl = $("recStats"); dl.textContent = ""; for (const [k, v] of rows) { const t = document.createElement("dt"), d = document.createElement("dd"); t.textContent = k; d.textContent = v; dl.append(t, d); } }
function recMsg(text, cls = "") { $("recMsg").textContent = text; $("recMsg").className = "msg " + cls; }
function stopLive(keep) {
  const f = $("live"); if (!keep) { f.hidden = true; f.src = "about:blank"; $("idle").hidden = false; }
  $("record").disabled = false; $("abort").disabled = true; S.recFrame = null;
}
$("record").addEventListener("click", () => {
  const params = normalize(S.params);
  if (!S.deployedSizes || !S.deployedSizes[params.flipAxis].has(params.size)) { recMsg(`${params.size} has no deployed ${params.flipAxis === "y" ? "2-D" : "3-D axis-enabled"} fountain product; wait for the build or choose an available size`, "err"); return; }
  stats([]); $("bar").value = 0; recMsg("starting the game page…");
  const f = $("live"); f.hidden = false; $("idle").hidden = true;
  f.src = "/" + toQuery(params);
  S.recFrame = f; S.rec0 = performance.now(); S.recParams = params;
  $("record").disabled = true; $("abort").disabled = false;
});
$("abort").addEventListener("click", () => { stopLive(false); recMsg("aborted"); });

window.addEventListener("message", (e) => {
  const m = e.data;
  if (e.origin !== location.origin || !m || m.type !== "loading-record" || !S.recFrame || e.source !== S.recFrame.contentWindow) return;
  const secs = ((performance.now() - S.rec0) / 1000).toFixed(0);
  if (m.stage === "tick") {
    $("bar").value = Math.min(95, 100 * (m.turns + (m.poured ? 0.3 : 0)) / (m.rounds + 1));
    recMsg(`${m.turns}/${m.rounds} turns, energy ${m.energy.toExponential(1)} (settles under ${m.threshold.toExponential(1)}), sim ${m.t.toFixed(1)}/${m.cap} s, ${secs} s elapsed`);
    stats([["frames in the state memory", String(m.frame)]]);
  } else if (m.stage === "progress") {
    recMsg(m.text); if (m.raw) stats([["raw stream", kb(m.raw)]]);
  } else if (m.stage === "error") {
    $("bar").value = 0; recMsg(m.text, "err");
    stats(m.raw ? [["raw stream", `${kb(m.raw)} (cap ${kb(RAW_CAP)})`]] : []);
    stopLive(true);
  } else if (m.stage === "done") {
    $("bar").value = 100;
    const s = m.stored || {};
    recMsg(s.duplicate ? "already in the library (same animation)" : `saved to the library${s.active ? " and active (the first one)" : " (not active)"}`, "ok");
    stats([["simulated ids", Number(m.sim_n).toLocaleString()], ["capture", m.visual_only ? "visual only (no physics replay)" : "state memory"], ["flip", `${Number(m.effective_flip_seconds).toFixed(2)} simulated s (requested ${Number(S.recParams.flipSeconds).toFixed(2)} s)`], ["raw stream", kb(m.raw)], ["octet codec", `${kb(s.bytes)}  (${(100 * s.bytes / m.raw).toFixed(1)}% of raw)`], ["frames x grains", `${s.frames} x ${s.grains} at ${s.fps} fps`],
           ["tried", (m.tried || []).map((t) => `${t.fps}fps/${t.grains}g/${t.bits}b=${kb(t.raw)}`).join("; ")], ...(m.capped ? [["ended by", "the duration cap"]] : [])]);
    stopLive(false);
    loadLibrary().then(() => { if (s.digest) { S.playing = s.digest; paintCards(); showScreen(); } });
  }
});

// ------------------------------------------------------------------ 3. the library
async function loadLibrary() {
  try {
    const r = await fetch("/api/loading_library", { cache: "no-store" });
    const body = await r.json();
    S.entries = body.entries || []; S.active = body.active;
    const c = body.caps || {};
    $("libNote").textContent = `${S.entries.length}/${c.max_entries} recordings, ${kb(c.used_bytes || 0)} of ${kb(c.max_bytes || 0)}`;
  } catch (error) { $("libMsg").textContent = `could not read the library: ${error.message}`; $("libMsg").className = "msg err"; }
  if (S.A && !S.entries.some((e) => e.digest === S.A)) S.A = null;
  if (S.B && !S.entries.some((e) => e.digest === S.B)) S.B = null;
  if (S.playing && !S.entries.some((e) => e.digest === S.playing)) S.playing = null;
  paintCards(); showScreen(); paintCompare();
}
const paramsLine = (p) => (p && p.chamber ? `${p.chamber.replace("sdf_", "")}, ${p.size}, ${p.kept_grains || p.grains} grains, ${p.bits} bit, ${p.fps} fps, ${p.turns} turns, grain x${Number(p.grainScale).toFixed(2)}${p.mix === "default" ? "" : `, mix ${p.mix}`}` : (p && p.migrated ? "migrated from the single-file store" : ""));
function paintCards() {
  const box = $("lib"); box.textContent = "";
  if (!S.entries.length) { const p = document.createElement("p"); p.className = "hint"; p.textContent = "Nothing recorded yet: players see the procedural hourglass. Record one above."; box.appendChild(p); return; }
  for (const e of S.entries) {
    const card = document.createElement("div"); card.className = "card" + (e.active ? " active" : "") + (e.digest === S.playing ? " sel" : "");
    const cv = document.createElement("canvas"); cv.width = 320; cv.height = 240; card.appendChild(cv);
    const meta = document.createElement("div"); meta.className = "meta";
    const title = document.createElement("div"); const b = document.createElement("b"); b.textContent = e.digest.slice(0, 8); title.appendChild(b);
    if (e.active) { const bd = document.createElement("span"); bd.className = "badge"; bd.textContent = "ACTIVE"; title.appendChild(bd); }
    const line1 = document.createElement("div"); line1.textContent = `${(e.date || "").replace("T", " ").replace("Z", "")}  ${e.frames} frames, ${e.grains} grains`;
    const line2 = document.createElement("div"); line2.textContent = `raw ${kb(e.raw_bytes)} / codec ${kb(e.bytes)} (${(100 * e.bytes / e.raw_bytes).toFixed(1)}%)`;
    const line3 = document.createElement("div"); line3.textContent = paramsLine(e.params);
    meta.append(title, line1, line2, line3); card.appendChild(meta);
    const btns = document.createElement("div"); btns.className = "btns";
    const mk = (text, fn, cls) => { const x = document.createElement("button"); x.textContent = text; if (cls) x.className = cls; x.addEventListener("click", fn); btns.appendChild(x); return x; };
    mk("Play", () => { S.playing = e.digest; paintCards(); showScreen(); window.scrollTo({ top: 0, behavior: "smooth" }); });
    mk(S.A === e.digest ? "A ✓" : "A", () => { S.A = e.digest; paintCompare(); paintCards(); });
    mk(S.B === e.digest ? "B ✓" : "B", () => { S.B = e.digest; paintCompare(); paintCards(); });
    mk("Make active", async () => { await ownerAction("/api/loading_library/activate", e, "made active"); }).disabled = e.active;
    mk("Delete", async () => { if (confirm(`Delete recording ${e.digest.slice(0, 8)}${e.active ? " (the ACTIVE one: players get the procedural hourglass until another is made active)" : ""}?`)) await ownerAction("/api/loading_library/delete", e, "deleted"); }, "danger");
    card.appendChild(btns); box.appendChild(card);
    thumb(e, cv);
    card.addEventListener("pointerenter", () => { S.hover = { digest: e.digest, cv, t0: performance.now() }; });
    card.addEventListener("pointerleave", () => { if (S.hover && S.hover.cv === cv) S.hover = null; });
  }
}
async function ownerAction(path, e, done) {
  const msg = $("libMsg");
  try { await owner(path, e.digest); msg.textContent = `${e.digest.slice(0, 8)} ${done}`; msg.className = "msg ok"; } catch (error) { msg.textContent = error.message; msg.className = "msg err"; return; }
  await loadLibrary();
}

// thumbnails: the entry decoded in the page (the same decoder the loading panel uses), drawn into a 2-D canvas; it plays while hovered
async function animOf(digest) {
  if (!S.thumbs.has(digest)) {
    S.thumbs.set(digest, fetch(`/api/loading_anim?v=${digest}`).then(async (r) => {
      if (!r.ok) throw new Error(String(r.status));
      const a = decodePacket(new Uint8Array(await r.arrayBuffer())), span = Math.max(a.bbox[1] - a.bbox[0], a.bbox[3] - a.bbox[2]);
      a.dx = (a.bbox[1] - a.bbox[0]) / span; a.dy = (a.bbox[3] - a.bbox[2]) / span; return a;
    }));
  }
  return S.thumbs.get(digest);
}
function drawThumb(a, cv, f) {
  const ctx = cv.getContext("2d"), w = cv.width, h = cv.height, R = 0.5 * Math.hypot(a.dx, a.dy), k = (0.46 * Math.min(w, h)) / R;
  ctx.fillStyle = "#101116"; ctx.fillRect(0, 0, w, h);
  ctx.save(); ctx.translate(w / 2, h / 2); ctx.rotate(angleOf(a.angle[f])); ctx.scale(k, k);
  ctx.translate(-a.dx / 2, -a.dy / 2);
  ctx.strokeStyle = "rgba(220,220,230,.55)"; ctx.lineWidth = 1.5 / k;
  for (const poly of a.outline) { ctx.beginPath(); for (let i = 0; i < poly.length; i += 2) ctx[i ? "lineTo" : "moveTo"]((poly[i] / 65535), (poly[i + 1] / 65535)); ctx.stroke(); }
  // y is up in the data; the panel's clip space has y up as well, so flip the canvas's y for the grains
  ctx.fillStyle = "#e0a85a";
  const sz = Math.max(1.4, 3.2 / k);
  for (let i = 0; i < a.radius.length; i++) { ctx.beginPath(); ctx.arc(a.x[f][i] / 65535, a.y[f][i] / 65535, Math.max(a.radius[i] / 65535, sz / k * 0.5), 0, 6.2832); ctx.fill(); }
  ctx.restore();
}
function thumb(e, cv) {
  animOf(e.digest).then((a) => { drawThumb(a, cv, Math.floor((a.x.length - a.seam) * 0.45)); }).catch(() => { const c = cv.getContext("2d"); c.fillStyle = "#555"; c.fillText("unreadable", 10, 20); });
}
(function hoverLoop() {
  const h = S.hover;
  if (h) animOf(h.digest).then((a) => { if (S.hover === h) { const [fa] = seam(Math.floor(((performance.now() - h.t0) / 1000) * a.fps), a.x.length, a.seam); drawThumb(a, h.cv, fa); } }).catch(() => {});
  setTimeout(() => requestAnimationFrame(hoverLoop), 60);
})();

// ------------------------------------------------------------------ compare
function paintCompare() {
  const show = !!(S.A || S.B);
  $("cmpSection").hidden = !show;
  for (const [id, key, label] of [["viewA", "A", "cmpA"], ["viewB", "B", "cmpB"]]) {
    const d = S[key], f = $(id);
    $(label).textContent = d ? `${key}: ${d.slice(0, 8)}${(S.entries.find((e) => e.digest === d) || {}).active ? " (active)" : ""}` : `${key}: pick a card`;
    if (!d) { if (f.dataset.d) { f.src = "about:blank"; delete f.dataset.d; } continue; }
    if (f.dataset.d !== d) { f.dataset.d = d; f.src = viewUrl(S.phase, d); }
  }
}
$("cmpClear").addEventListener("click", () => { S.A = S.B = null; paintCompare(); paintCards(); });
$("cmpSync").addEventListener("click", () => { for (const [id, key] of [["viewA", "A"], ["viewB", "B"]]) if (S[key]) post($(id), { anim: S[key] }); });

buildPhases(); buildControls(); loadSizes(); loadLibrary();
