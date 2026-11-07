// REPLAY LAYERS ("ghosts"): several recorded runs of the fountain drawn at once with the live one (pure logic: no GPU, no DOM,
// so node can hold it in a test).
//
// A recording frame (recording.js, compute.js) is the packed points of one snapshot: 5 words per displayed id,
// [x, y, z, rgba8 (as bits), meta], and the id IS the index (times the display stride) -- stable across steps and across
// runs of the same law, which is what lets runs be laid over each other.  This module only READS those frames; the
// recorder's format and capture path are untouched.
//
// * IDENTITY.  A grain in a layer is (layer id, grain id); `gid` packs the pair.  Two layers' copies of one grain are
//   JOINED on the grain id (`joinIds`), never on position.
// * STEP MAPPING.  All layers play on the live run's clock: the live run's displayed frame index (= its step count / snapEvery)
//   picks each layer's frame (`stepMap`); a layer loops, holds its last frame, or bounces at its end.
// * MEMORY.  A layer keeps the REDUCED representation of each frame: x, y, z quantised to 16 bits inside the frame's own
//   bounds and the meta word (liveness, at-rest flag, radius) in 16 bits = 8 bytes per grain (the full frame is 20); colour
//   is not kept (a ghost takes the live copy of the same grain id, or the layer's tint).  `LayerSet` holds the total under a
//   byte cap.
// * COMPOSITION.  "behind": each layer on its own parallel plane, `depth` apart behind the live one; "single": every layer in
//   the live plane, told apart by tint / opacity.  "diverge" colours a ghost grain by the distance to its live copy.
// * MAKING.  `makeGhosts` runs K short extra runs through the existing recorder (a second ComputeRuntime per run, from the
//   start, with a seeded perturbation of runtime inputs), reduces their frames and returns the layers.
export const PT_WORDS = 5;
export const REDUCED_WORDS = 4;                              // x, y, z, meta as Uint16
export const META_SCALE = 64;                                // meta (flag * 100 + radius * 1000) is below ~1000: 16 bits at 1/64
export const MAX_LAYERS = 6;
export const ID_BITS = 24;                                   // a layer's grain ids stay below 2^24 (the law's own limit on the step/id arithmetic)
export const HEADER_BYTES = 24;                              // the per-frame bounds (6 floats)

// ---------------------------------------------------------------- identity
export const gid = (layer, id) => layer * 2 ** ID_BITS + id;
export const gidLayer = (g) => Math.floor(g / 2 ** ID_BITS);
export const gidId = (g) => g % 2 ** ID_BITS;
// the grain ids a frame's rows carry: row i is id i * stride
export const rowIds = (n, stride = 1) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = i * stride; return a; };

// Join two id lists on the id: returns [rowsA, rowsB] (equal lengths) of the rows holding the same id.  Equal lists (the
// usual case: two runs of one law at one size) take the identity fast path.
export function joinIds(a, b) {
  const n = Math.min(a.length, b.length);
  let same = a.length === b.length;
  for (let i = 0; same && i < n; i++) same = a[i] === b[i];
  if (same) { const r = new Uint32Array(n); for (let i = 0; i < n; i++) r[i] = i; return [r, r]; }
  const at = new Map();
  for (let i = 0; i < b.length; i++) at.set(b[i], i);
  const ra = [], rb = [];
  for (let i = 0; i < a.length; i++) { const j = at.get(a[i]); if (j !== undefined) { ra.push(i); rb.push(j); } }
  return [Uint32Array.from(ra), Uint32Array.from(rb)];
}

// ---------------------------------------------------------------- the reduced frame
const isLive = (meta) => meta >= 50;

// A full packed frame (Float32Array, 5 words per id) -> { n, bounds: Float32Array(6), q: Uint16Array(4n) }.
export function reduceFrame(frame, n = frame.length / PT_WORDS) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    if (!isLive(frame[5 * i + 4])) continue;
    for (let c = 0; c < 3; c++) { const v = frame[5 * i + c]; if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
  }
  for (let c = 0; c < 3; c++) if (!(hi[c] >= lo[c])) { lo[c] = 0; hi[c] = 0; }
  const bounds = Float32Array.of(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]), q = new Uint16Array(REDUCED_WORDS * n);
  const span = [0, 1, 2].map((c) => (hi[c] > lo[c] ? 65535 / (hi[c] - lo[c]) : 0));
  for (let i = 0; i < n; i++) {
    const m = frame[5 * i + 4];
    if (!isLive(m)) continue;                                // q stays 0: meta 0 = not drawn
    for (let c = 0; c < 3; c++) q[4 * i + c] = Math.round((frame[5 * i + c] - bounds[c]) * span[c]);
    q[4 * i + 3] = Math.max(1, Math.min(65535, Math.round(m * META_SCALE)));
  }
  return { n, bounds, q };
}
export const reducedBytes = (n) => REDUCED_WORDS * 2 * n + HEADER_BYTES;
export const fullBytes = (n) => PT_WORDS * 4 * n;

export function positionOf(r, i, out = [0, 0, 0]) {
  const b = r.bounds;
  for (let c = 0; c < 3; c++) out[c] = b[c] + (r.q[4 * i + c] / 65535) * (b[3 + c] - b[c]);
  return out;
}
export const metaOf = (r, i) => r.q[4 * i + 3] / META_SCALE;

// ---------------------------------------------------------------- the shared clock
// The layer's frame for the live run's displayed frame `shown`, or -1 when the layer has not started yet.  A layer holds
// `count` frames, the first of which was the run's frame `first`.
export function stepMap(shown, { first, count, mode = "loop" }) {
  if (!(count > 0) || shown == null) return -1;
  const rel = shown - first;
  if (rel < 0) return -1;
  if (mode === "hold") return Math.min(rel, count - 1);
  if (mode === "bounce") {
    if (count === 1) return 0;
    const period = 2 * (count - 1), p = rel % period;
    return p < count ? p : period - p;
  }
  return rel % count;                                        // loop
}

// ---------------------------------------------------------------- divergence
// The distance between the two copies of a grain (law units).  `scale` is the distance that reads as fully diverged.
export const divergenceOf = (ax, ay, az, bx, by, bz) => Math.hypot(ax - bx, ay - by, az - bz);
export const divergenceUnit = (d, scale) => (scale > 0 ? Math.min(1, d / scale) : 0);
// colour ramp for a divergence u in [0, 1]: cool blue (together) -> cyan -> yellow -> red (far apart); bytes
export function divergenceRamp(u) {
  const stops = [[40, 90, 255], [30, 220, 230], [255, 225, 40], [255, 50, 40]], f = Math.max(0, Math.min(1, u)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(f)), t = f - i;
  return stops[i].map((v, c) => Math.round(v + (stops[i + 1][c] - v) * t));
}
const packRGBA = (r, g, b, a) => (((a & 255) << 24) | ((b & 255) << 16) | ((g & 255) << 8) | (r & 255)) >>> 0;
export function parseHex(hex, fallback = [255, 255, 255]) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ""));
  if (!m) return fallback;
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

// Fill `out` (a Float32Array of 5 * reduced.n words, the drawing format) with a ghost layer's frame.
//   live      the live run's current packed frame (or null): the colour of the same grain id, and the divergence partner
//   zOffset   added to every grain's z (the layer's plane)
//   tint      [r, g, b] bytes; tintAmount 0..1 how far the grain's own colour is pulled to it; alpha 0..1 the layer's opacity
//   diverge   null, or { scale }: colour by the distance to the live copy (joined on the grain id)
// Returns the divergence statistics { joined, mean, max, over } (zeros without a live frame / diverge).
export function buildGhostFrame(out, reduced, live, { zOffset = 0, tint = [255, 255, 255], tintAmount = 0.7, alpha = 0.5, diverge = null, stride = 1, liveStride = stride } = {}) {
  const n = reduced.n, q = reduced.q, b = reduced.bounds, bits = new Uint32Array(out.buffer, out.byteOffset, out.length);
  const sx = (b[3] - b[0]) / 65535, sy = (b[4] - b[1]) / 65535, sz = (b[5] - b[2]) / 65535;
  const a8 = Math.max(0, Math.min(255, Math.round(alpha * 255)));
  const liveN = live ? Math.floor(live.length / PT_WORDS) : 0, liveBits = live ? new Uint32Array(live.buffer, live.byteOffset, live.length) : null;
  // the join: row i of the layer is id i * stride; the live frame's row j is id j * liveStride
  const same = live && stride === liveStride;
  let rowOf = null;                                        // layer row -> live row, joined on the grain id (identity when the strides agree)
  if (live && !same) {
    const [ra, rb] = joinIds(rowIds(n, stride), rowIds(liveN, liveStride));
    rowOf = new Int32Array(n).fill(-1);
    for (let k = 0; k < ra.length; k++) rowOf[ra[k]] = rb[k];
  }
  const stats = { joined: 0, mean: 0, max: 0, over: 0 };
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const m = q[4 * i + 3];
    if (!m) { out[5 * i + 4] = 0; out[5 * i] = 0; out[5 * i + 1] = 0; out[5 * i + 2] = 0; bits[5 * i + 3] = 0; continue; }
    const x = b[0] + q[4 * i] * sx, y = b[1] + q[4 * i + 1] * sy, z = b[2] + q[4 * i + 2] * sz;
    out[5 * i] = x; out[5 * i + 1] = y; out[5 * i + 2] = z + zOffset; out[5 * i + 4] = m / META_SCALE;
    const j = !live ? -1 : same ? i : rowOf[i];
    const hasLive = live && j >= 0 && j < liveN && live[5 * j + 4] >= 50;
    let r = 255, g = 255, bl = 255;
    if (hasLive) { const c = liveBits[5 * j + 3]; r = c & 255; g = (c >> 8) & 255; bl = (c >> 16) & 255; }
    if (diverge && hasLive) {
      const d = divergenceOf(x, y, z, live[5 * j], live[5 * j + 1], live[5 * j + 2]);
      stats.joined++; sum += d; if (d > stats.max) stats.max = d; if (d > diverge.scale * 0.5) stats.over++;
      [r, g, bl] = divergenceRamp(divergenceUnit(d, diverge.scale));
    } else {
      r = Math.round(r + (tint[0] - r) * tintAmount); g = Math.round(g + (tint[1] - g) * tintAmount); bl = Math.round(bl + (tint[2] - bl) * tintAmount);
    }
    bits[5 * i + 3] = packRGBA(r, g, bl, a8);
  }
  if (stats.joined) stats.mean = sum / stats.joined;
  return stats;
}

// ---------------------------------------------------------------- layers and their memory
export class ReplayLayer {
  // frames: reduced frames of consecutive run frames first, first + 1, ...
  constructor({ id, name = "", seed = null, first = 1, frames = [], stride = 1, mode = null, source = "made", perturbation = null }) {
    this.id = id; this.name = name; this.seed = seed; this.first = first; this.frames = frames; this.stride = stride;
    this.mode = mode; this.source = source; this.perturbation = perturbation; this.lastStats = null;
  }
  get count() { return this.frames.length; }
  get n() { return this.frames.length ? this.frames[0].n : 0; }
  get bytes() { return this.frames.reduce((s, f) => s + f.q.byteLength + f.bounds.byteLength, 0); }
  // the layer's frame for the live run's displayed frame `shown` (null: not started / empty)
  frameAt(shown, mode) {
    const k = stepMap(shown, { first: this.first, count: this.count, mode: this.mode || mode });
    return k < 0 ? null : this.frames[k];
  }
  // the grain `id` of this layer as a global identity
  gid(id) { return gid(this.id, id); }
}

// The layers under ONE byte cap.  Slots are the layers list's rows (a settings row i belongs to slot i).
export class LayerSet {
  constructor(capBytes = 48 * 1048576, slots = MAX_LAYERS) { this.capBytes = capBytes; this.slots = new Array(slots).fill(null); }
  get bytes() { return this.slots.reduce((s, l) => s + (l ? l.bytes : 0), 0); }
  get used() { return this.slots.filter(Boolean).length; }
  freeSlot() { return this.slots.findIndex((l) => !l); }
  has(i) { return !!this.slots[i]; }
  // Put `layer` in a slot (the first free one, or `slot`); refuses (returns -1) when it would pass the cap.
  add(layer, slot = this.freeSlot()) {
    if (slot < 0 || slot >= this.slots.length) return -1;
    const old = this.slots[slot] ? this.slots[slot].bytes : 0;
    if (this.bytes - old + layer.bytes > this.capBytes) return -1;
    layer.id = slot + 1;                                     // layer 0 is the live run
    this.slots[slot] = layer;
    return slot;
  }
  remove(i) { const l = this.slots[i]; this.slots[i] = null; return l; }
  clear() { this.slots.fill(null); }
  // room (bytes) left for a new layer
  get room() { return Math.max(0, this.capBytes - this.bytes); }
}

// How long a ghost run can be so that `k` layers of `n` ids fit in `room` bytes: { frames, perLayer, total, truncated }.
export function planGhosts({ room, k, n, snapEvery, dt, seconds }) {
  const wanted = Math.max(1, Math.ceil(seconds / (snapEvery * dt))), per = reducedBytes(n);
  const fit = Math.floor(room / Math.max(1, k) / per), frames = Math.max(0, Math.min(wanted, fit));
  return { frames, wanted, perFrame: per, perLayer: frames * per, total: frames * per * k, truncated: frames < wanted };
}

// ---------------------------------------------------------------- perturbation (a seed -> a deterministic change of runtime inputs)
function mulberry32(a) { return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = (rnd) => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
// The fountain's law has no seed (its noise is a pure hash of id and step), so runs differ only by their runtime inputs: a
// ghost's seed perturbs the case angle (radians) and the nozzle speed (relative).  Seed 0 is the control: no change at all.
export function perturbation(seed, amp = 1, { angleDeg = 0.5, nozzleRel = 0.01 } = {}) {
  if (!seed || !amp) return { dtheta: 0, dnozzle: 0 };
  const rnd = mulberry32((seed * 2654435761) >>> 0);
  return { dtheta: gauss(rnd) * amp * angleDeg * Math.PI / 180, dnozzle: gauss(rnd) * amp * nozzleRel };
}

// ---------------------------------------------------------------- settings (the page specs; share links carry these, not the recordings)
export const TINTS = ["#4fd8ff", "#ff5fd2", "#ffc23d", "#7dff7a", "#a98bff", "#ff8a4d"];
export const COMPOSITIONS = ["off", "behind", "single"], LOOP_MODES = ["loop", "hold", "bounce"];
export const DEFAULT_SETTINGS = () => ({
  comp: "off", depth: 0.08, diverge: false, dscale: 0.05, mode: "loop", amount: 0.75, k: 3, secs: 4, amp: 1,
  slots: TINTS.map((tint) => ({ on: true, alpha: 0.45, tint })),
});
const clamp = (v, lo, hi, d) => { v = Number(v); return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : d; };
// ids of the settings specs: ghosts_comp, ghosts_depth, ... and per slot ghosts<i>_on, ghosts<i>_alpha, ghosts<i>_tint (i from 1)
export function applySetting(s, id, value) {
  let m;
  if ((m = /^ghosts([1-9])_(on|alpha|tint)$/.exec(id))) {
    const slot = s.slots[Number(m[1]) - 1];
    if (!slot) return s;
    if (m[2] === "on") slot.on = value === true || value === "true" || value === 1 || value === "1";
    else if (m[2] === "alpha") slot.alpha = clamp(value, 0, 1, slot.alpha);
    else slot.tint = /^#[0-9a-f]{6}$/i.test(String(value)) ? String(value).toLowerCase() : slot.tint;
    return s;
  }
  const d = DEFAULT_SETTINGS();
  switch (id) {
    case "ghosts_comp": s.comp = COMPOSITIONS.includes(value) ? value : d.comp; break;
    case "ghosts_mode": s.mode = LOOP_MODES.includes(value) ? value : d.mode; break;
    case "ghosts_depth": s.depth = clamp(value, 0, 0.5, d.depth); break;
    case "ghosts_diverge": s.diverge = value === true || value === "true" || value === 1 || value === "1"; break;
    case "ghosts_dscale": s.dscale = clamp(value, 0.002, 0.5, d.dscale); break;
    case "ghosts_amount": s.amount = clamp(value, 0, 1, d.amount); break;
    case "ghosts_k": s.k = Math.round(clamp(value, 1, MAX_LAYERS, d.k)); break;
    case "ghosts_secs": s.secs = clamp(value, 0.5, 30, d.secs); break;
    case "ghosts_amp": s.amp = clamp(value, 0, 4, d.amp); break;
    default: break;
  }
  return s;
}
export function settingValue(s, id) {
  let m;
  if ((m = /^ghosts([1-9])_(on|alpha|tint)$/.exec(id))) { const slot = s.slots[Number(m[1]) - 1]; return slot ? slot[m[2]] : undefined; }
  return { ghosts_comp: s.comp, ghosts_mode: s.mode, ghosts_depth: s.depth, ghosts_diverge: s.diverge, ghosts_dscale: s.dscale, ghosts_amount: s.amount, ghosts_k: s.k, ghosts_secs: s.secs, ghosts_amp: s.amp }[id];
}
export function sanitizeSettings(raw) {
  const s = DEFAULT_SETTINGS();
  if (!raw || typeof raw !== "object") return s;
  for (const key of ["comp", "mode", "depth", "diverge", "dscale", "amount", "k", "secs", "amp"]) if (key in raw) applySetting(s, "ghosts_" + key, raw[key]);
  (Array.isArray(raw.slots) ? raw.slots : []).slice(0, MAX_LAYERS).forEach((slot, i) => {
    if (slot) for (const key of ["on", "alpha", "tint"]) if (key in slot) applySetting(s, `ghosts${i + 1}_${key}`, slot[key]);
  });
  return s;
}

// What to draw this frame: the enabled layers with their frame, in drawing order (the farthest first, so the near ones and the
// live run, drawn after, sit on top), each with its plane offset.  `width` is the view's width in law units.
export function planDraw(set, settings, shown, width) {
  if (settings.comp === "off" || shown == null) return [];
  const out = [];
  set.slots.forEach((layer, i) => {
    const row = settings.slots[i];
    if (!layer || !row || !row.on || !(row.alpha > 0)) return;
    const frame = layer.frameAt(shown, settings.mode);
    if (!frame) return;
    out.push({ slot: i, layer, frame, alpha: row.alpha, tint: parseHex(row.tint),
               zOffset: settings.comp === "behind" ? -settings.depth * width * (out.length + 1) : 0 });
  });
  return out.reverse();                                     // the nearest plane (smallest offset) last
}

// ---------------------------------------------------------------- saving a layer (a file the page can write; the share link does not carry it)
export function serializeLayer(layer) {
  const head = JSON.stringify({ v: 1, name: layer.name, seed: layer.seed, first: layer.first, stride: layer.stride, n: layer.n, count: layer.count, source: layer.source });
  const hb = new TextEncoder().encode(head), body = layer.frames.reduce((s, f) => s + f.bounds.byteLength + f.q.byteLength, 0);
  const out = new Uint8Array(4 + hb.length + body), dv = new DataView(out.buffer);
  dv.setUint32(0, hb.length, true); out.set(hb, 4);
  let at = 4 + hb.length;
  for (const f of layer.frames) { out.set(new Uint8Array(f.bounds.buffer, f.bounds.byteOffset, f.bounds.byteLength), at); at += f.bounds.byteLength; out.set(new Uint8Array(f.q.buffer, f.q.byteOffset, f.q.byteLength), at); at += f.q.byteLength; }
  return out;
}
export function deserializeLayer(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), hl = dv.getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + hl)));
  let at = 4 + hl;
  const frames = [];
  for (let k = 0; k < head.count; k++) {
    const bounds = new Float32Array(bytes.slice(at, at + 24).buffer); at += 24;
    const q = new Uint16Array(bytes.slice(at, at + head.n * 8).buffer); at += head.n * 8;
    frames.push({ n: head.n, bounds, q });
  }
  return new ReplayLayer({ id: 0, name: head.name, seed: head.seed, first: head.first, stride: head.stride, frames, source: head.source });
}

// ---------------------------------------------------------------- layers from the live recording
// The live recorder's window (recording.js: frames lo..hi) as one layer.  Under `room` bytes the NEWEST frames are kept.
export function layerFromRecording(tape, { name = "recorded", stride = 1, room = Infinity } = {}) {
  if (!tape || tape.hi < tape.lo) return null;
  const n = tape.frameBytes / (PT_WORDS * 4), per = reducedBytes(n), want = tape.hi - tape.lo + 1;
  const count = Math.min(want, Math.floor(room / per));
  if (count < 2) return null;
  const first = tape.hi - count + 1, frames = [];
  for (let k = first; k <= tape.hi; k++) { const f = tape.get(k); if (!f) return null; frames.push(reduceFrame(f, n)); }
  return new ReplayLayer({ id: 0, name, first, frames, stride, source: "recorded" });
}

// ---------------------------------------------------------------- making ghosts (K extra runs through the existing recorder)
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// `live` is the page's ComputeRuntime; `makeRuntime()` a fresh one (the same class).  Each ghost is a second runtime that re-runs
// the live run's settings from its start (frame 1), with the perturbation of its seed, until `frames` snapshots are recorded; the
// frames are reduced and the runtime freed before the next one starts.  Resolves to the layers (not yet in a set).
export async function makeGhosts({ live, makeRuntime, seeds, frames, amp = 1, stride = 1, onProgress = () => {}, onLayer = () => {}, shouldStop = () => false, sleepMs = 8 }) {
  const out = [], n = live.tape.frameBytes / (PT_WORDS * 4);
  for (let g = 0; g < seeds.length && !shouldStop(); g++) {
    const seed = seeds[g], p = perturbation(seed, amp), rt = makeRuntime();
    try {
      rt.setRecording(true);
      rt.capBytes = (frames + 64) * live.tape.frameBytes;      // the window holds every frame of the run (+ the batches in flight)
      rt.inheritFrom(live, { caseTiltAdd: p.dtheta, nozzleScale: 1 + p.dnozzle });
      rt.sync({ ...live.spec, speed: 0, uniforms: rt.perturbedUniforms(live.spec.uniforms, p.dnozzle) });
      for (let waited = 0; !rt.ready; waited += sleepMs) {
        if (rt.error) throw new Error(rt.error);
        if (waited > 180000) throw new Error("the ghost run did not start");
        await sleep(sleepMs);
      }
      rt.tape.keyEvery = 1e15;                                 // no keyframes: a ghost is displayed, never continued
      while (rt.tape.hi < frames && !shouldStop()) {
        if (rt.error) throw new Error(rt.error);
        onProgress({ ghost: g, of: seeds.length, frame: Math.max(0, rt.tape.hi), frames });
        await sleep(sleepMs);
      }
      rt.running = false;
      for (let w = 0; rt.inflight > 0 && w < 2000; w++) await sleep(2);
      if (shouldStop()) break;
      const lo = rt.tape.lo, list = [];
      if (lo > 1) throw new Error(`the ghost window slid to frame ${lo}`);
      for (let k = 1; k <= frames; k++) { const f = rt.tape.get(k); if (!f) break; list.push(reduceFrame(f, n)); }
      const layer = new ReplayLayer({ id: 0, name: `seed ${seed}`, seed, first: 1, frames: list, stride, source: "made", perturbation: p });
      out.push(layer); onLayer(layer, g);
    } finally { rt.dispose(); }
    onProgress({ ghost: g + 1, of: seeds.length, frame: frames, frames });
  }
  return out;
}
