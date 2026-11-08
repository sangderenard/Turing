// The loading animation (pure: no GPU, no DOM, so node holds it in a test).  Formats MLA1/MLA2 and their loop seam are documented in
// loading_anim.py, whose codec this twins byte for byte.  The file the page FETCHES is that stream through the octet codec
// (octet_codec.js); `filterTape` is how the stream is MADE: a reduced filter of the compute runtime's state memory
// (recording.js `Recording`).  For runs whose complete physics tape exceeds its sliding window, VisualTape keeps
// packed points at the requested output cadence; it cannot resume or replay the law.
import { decompressOctets } from "./octet_codec.js";
import { decodeMeta } from "./particle_styles.js";

export const MAGIC = "MLA2", HEADER_BYTES = 4 + 1 + 1 + 4 + 4 + 2 + 2 + 16;
const LEGACY_HEADER_BYTES = 30;
const TAU = 2 * Math.PI;

// A position-filter source for large simulations.  Frames are copied from the same packed point snapshots
// handed to Recording.push; the full physics state and keyframes remain owned by the compute runtime.
export class VisualTape {
  constructor({ sourceSnap, stride, sourceCount }) {
    this.snapEvery = sourceSnap * stride; this.stride = stride; this.sourceSnap = sourceSnap;
    this.sourceCount = sourceCount;
    this.frames = []; this.sourceFrames = []; this.inputs = []; this.lo = 1; this.hi = 0;
    this.ids = null; this.bytes = 0;
  }
  get(k) { return this.frames[k - 1] || null; }
  stepOf(k) { return this.sourceFrames[k - 1] * this.sourceSnap; }
  push(sourceFrame, data) {
    if (this.sourceFrames.length && sourceFrame <= this.sourceFrames.at(-1)) return;
    if (!this.ids) {
      const live = [];
      for (let i = 0; i < this.sourceCount; i++) if (data[5 * i + 4] >= 50) live.push(i);
      this.ids = live;
      if (!this.ids.length) throw new Error("visual recording: no live grains");
    }
    const bytes = this.ids.length * 20;
    const copy = new Float32Array(this.ids.length * 5), words = new Uint32Array(copy.buffer);
    const sourceWords = new Uint32Array(data.buffer, data.byteOffset, data.length);
    this.ids.forEach((id, j) => { for (let w = 0; w < 5; w++) words[5 * j + w] = sourceWords[5 * id + w]; });
    this.frames.push(copy); this.sourceFrames.push(sourceFrame); this.hi = this.frames.length; this.bytes += bytes;
  }
}

// ------------------------------------------------------------------ the stream (raw MLA1)
const planar = (u16) => {
  const n = u16.length, out = new Uint8Array(2 * n);
  for (let i = 0; i < n; i++) { out[i] = u16[i] & 255; out[n + i] = u16[i] >> 8; }
  return out;
};
const unplanar = (bytes, at, n) => {
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) out[i] = bytes[at + i] | (bytes[at + n + i] << 8);
  return out;
};

// anim: { fps, seam, bbox:[x0,x1,y0,y1], radius:Uint16Array(n), angle:Uint16Array(F), outline:[Uint16Array(2k)...] (x,y pairs), x:Uint16Array[F], y:Uint16Array[F] }
export function encodeStream(anim) {
  const F = anim.x.length, n = anim.x[0].length, parts = [];
  const head = new DataView(new ArrayBuffer(HEADER_BYTES));
  for (let i = 0; i < 4; i++) head.setUint8(i, MAGIC.charCodeAt(i));
  if (n > 0xffffffff || F > 0xffffffff) throw new Error("animation: counts exceed the stream format");
  head.setUint8(4, 2); head.setUint8(5, anim.fps); head.setUint32(6, n, true); head.setUint32(10, F, true);
  head.setUint16(14, anim.seam, true); head.setUint16(16, (anim.outline || []).length, true);
  anim.bbox.forEach((v, i) => head.setFloat32(18 + 4 * i, v, true));
  parts.push(new Uint8Array(head.buffer), planar(anim.radius), planar(anim.angle));
  for (const poly of anim.outline || []) {
    const k = poly.length / 2, xs = new Uint16Array(k), ys = new Uint16Array(k);
    for (let i = 0; i < k; i++) { xs[i] = poly[2 * i]; ys[i] = poly[2 * i + 1]; }
    parts.push(new Uint8Array([k & 255, k >> 8]), planar(xs), planar(ys));
  }
  parts.push(planar(anim.x[0]), planar(anim.y[0]));
  for (let f = 1; f < F; f++) {
    for (const plane of [anim.x, anim.y]) {
      const d = new Uint16Array(n);
      for (let i = 0; i < n; i++) d[i] = (plane[f][i] - plane[f - 1][i]) & 0xffff;
      parts.push(planar(d));
    }
  }
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0; for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

export function decodeStream(bytes) {
  if (bytes.length < LEGACY_HEADER_BYTES) throw new Error("animation: shorter than the header");
  const head = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const legacy = String.fromCharCode(...bytes.subarray(0, 4)) === "MLA1" && head.getUint8(4) === 1;
  if (!legacy && (String.fromCharCode(...bytes.subarray(0, 4)) !== MAGIC || head.getUint8(4) !== 2)) throw new Error("animation: unknown stream format");
  if (!legacy && bytes.length < HEADER_BYTES) throw new Error("animation: shorter than the header");
  const fps = head.getUint8(5), n = legacy ? head.getUint16(6, true) : head.getUint32(6, true);
  const F = legacy ? head.getUint16(8, true) : head.getUint32(10, true);
  const seam = head.getUint16(legacy ? 10 : 14, true), P = head.getUint16(legacy ? 12 : 16, true);
  const bbox = [0, 1, 2, 3].map((i) => head.getFloat32((legacy ? 14 : 18) + 4 * i, true));
  if (!fps || !n || F < 2 || P > 8 || seam * 2 > F || !(bbox[1] > bbox[0] && bbox[3] > bbox[2])) throw new Error("animation: header field out of range");
  let at = legacy ? LEGACY_HEADER_BYTES : HEADER_BYTES;
  const take = (count) => { if (at + 2 * count > bytes.length) throw new Error("animation: truncated"); const a = unplanar(bytes, at, count); at += 2 * count; return a; };
  const radius = take(n), angle = take(F), outline = [];
  for (let p = 0; p < P; p++) {
    if (at + 2 > bytes.length) throw new Error("animation: truncated");
    const k = bytes[at] | (bytes[at + 1] << 8); at += 2;
    const xs = take(k), ys = take(k), poly = new Uint16Array(2 * k);
    for (let i = 0; i < k; i++) { poly[2 * i] = xs[i]; poly[2 * i + 1] = ys[i]; }
    outline.push(poly);
  }
  const x = [take(n)], y = [take(n)];
  for (let f = 1; f < F; f++) {
    for (const [plane, prev] of [[x, x[f - 1]], [y, y[f - 1]]]) {
      const d = take(n), cur = new Uint16Array(n);
      for (let i = 0; i < n; i++) cur[i] = (prev[i] + d[i]) & 0xffff;
      plane.push(cur);
    }
  }
  if (at !== bytes.length) throw new Error("animation: trailing bytes");
  return { fps, seam, bbox, radius, angle, outline, x, y };
}

// what the server serves: the stream through the octet codec
export const decodePacket = (packet) => decodeStream(decompressOctets(packet));

// ------------------------------------------------------------------ the loop
// Playing position p of a loop of (F - S) frames -> [frame a, frame b, weight of b]; see loading_anim.py `seam`.
export function seam(p, frames, overlap) {
  const period = frames - overlap;
  p = ((p % period) + period) % period;
  if (p < overlap) return [p + period, p, (p + 0.5) / overlap];
  return [p, p, 0];
}

export const angleOf = (u16) => (u16 / 65536) * TAU;

// ------------------------------------------------------------------ the filter of the state memory
// The recording (recording.js) holds, per frame k in [lo, hi], the packed points (particle_styles.js PACKING: x, y, z, rgba8, meta; 5 words),
// and an input log of the runtime law inputs (gravity) by step.  Frame k is the state after k * snapEvery steps.
//
//   tape       a Recording
//   from, to   the frames to keep (inclusive); the loop is this span
//   grains     every grain alive in every kept frame is retained
//   bits       position bits kept (the stream stays u16: the value is shifted up, so the low bits are zero and the codec drops them)
//   fps        output frames per second of simulated time; `law.dt` is the law's step in seconds
//   turnSeconds  how long the view takes to follow a turn of the case (the drawing eases; the law's gravity changed at once)
const outlineBox = (lines) => {
  const pts = lines.flat();
  return pts.length ? [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))] : null;
};
export function filterTape(tape, { from, to, fps = 12, bits = 12, dt, outline = [], view, turnSeconds = 0.45, seamSeconds = 1.0, angleAtStep = null }) {
  from = Math.max(from, tape.lo); to = Math.min(to, tape.hi);
  if (to <= from) throw new Error("filterTape: nothing recorded between the frames");
  const stride = Math.max(1, Math.round(1 / (fps * dt * tape.snapEvery)));          // recorded frames per kept frame
  const frames = []; for (let k = from; k <= to; k += stride) frames.push(k);
  const first = tape.get(from), total = Math.floor(first.length / 5);
  const alive = (data, i) => data[5 * i + 4] >= 50;
  // the grains alive in every kept frame, with no count thinning
  const live = [];
  for (let i = 0; i < total; i++) { let ok = true; for (const k of frames) if (!alive(tape.get(k), i)) { ok = false; break; } if (ok) live.push(i); }
  const ids = live;
  if (!ids.length) throw new Error("filterTape: no grain is alive in every kept frame");
  const box = view || outlineBox(outline) || [Infinity, -Infinity, Infinity, -Infinity];
  if (!view) for (const k of frames) { const d = tape.get(k); for (const i of ids) { box[0] = Math.min(box[0], d[5 * i]); box[1] = Math.max(box[1], d[5 * i]); box[2] = Math.min(box[2], d[5 * i + 1]); box[3] = Math.max(box[3], d[5 * i + 1]); } }
  const bbox = [box[0], box[1], box[2], box[3]].map(Math.fround), span = Math.max(bbox[1] - bbox[0], bbox[3] - bbox[2]);
  const top = (1 << bits) - 1, q = (v, lo) => Math.max(0, Math.min(top, Math.round(((v - lo) / span) * top))) << (16 - bits);
  const x = [], y = [];
  for (const k of frames) {
    const d = tape.get(k), xs = new Uint16Array(ids.length), ys = new Uint16Array(ids.length);
    ids.forEach((i, j) => { xs[j] = q(d[5 * i], bbox[0]); ys[j] = q(d[5 * i + 1], bbox[2]); });
    x.push(xs); y.push(ys);
  }
  const radius = Uint16Array.from(ids, (i) => Math.max(0, Math.min(65535, Math.round((decodeMeta(first[5 * i + 4]).radius / span) * 65535))));
  const poly = outline.map((line) => { const a = new Uint16Array(2 * line.length); line.forEach(([px, py], i) => { a[2 * i] = q(px, bbox[0]); a[2 * i + 1] = q(py, bbox[2]); }); return a; });
  // the case's view angle per kept frame: the logged gravity direction (R(a) of (0, -G) is (G sin a, -G cos a)), unwrapped so a turn keeps
  // its direction, eased over `turnSeconds` of simulated time after the step it was logged at
  let target = 0, shown = 0, last = null, lastStep = null, turnStart = 0, turnFrom = 0;
  const angle = new Uint16Array(frames.length);
  const events = tape.inputs.filter((e) => "grav_x" in e.values && "grav_y" in e.values);
  frames.forEach((k, j) => {
    const step = tape.stepOf(k);
    if (angleAtStep) { const a = angleAtStep(step); angle[j] = Math.round((((-a % TAU) + TAU) % TAU) / TAU * 65536) & 0xffff; return; }
    let e = null; for (const ev of events) { if (ev.step <= step) e = ev; else break; }
    if (e && e.step !== lastStep) {
      lastStep = e.step;
      const raw = Math.atan2(e.values.grav_x, -e.values.grav_y);
      if (last === null) { target = raw; shown = raw; turnFrom = raw; turnStart = e.step; }
      else {
        const delta = (((raw - last) % TAU) + TAU) % TAU;                                    // turn the same way round, never back
        if (delta > 1e-6 && TAU - delta > 1e-6) { turnFrom = shown; target += delta; turnStart = e.step; }
      }
      last = raw;
    }
    const u = Math.min(1, Math.max(0, ((step - turnStart) * dt) / turnSeconds)), s = u * u * (3 - 2 * u);
    shown = turnFrom + (target - turnFrom) * s;
    angle[j] = Math.round((((-shown % TAU) + TAU) % TAU) / TAU * 65536) & 0xffff;             // the view turns the picture the OPPOSITE way
  });
  return { fps, seam: Math.min(Math.floor(frames.length / 2), Math.round(seamSeconds * fps)), bbox, radius, angle, outline: poly, x, y, grains: ids.length };
}
