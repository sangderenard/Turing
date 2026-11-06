// The run's RECORDING and the cursor that plays it (pure logic: no GPU, no DOM, so node can hold it in a test).
//
// The compute runtime steps the law as fast as it can and hands every `snapEvery`-th step's packed points to
// `Recording.push`.  The recording is a SLIDING WINDOW of those snapshots under a memory cap (`capBytes`): the oldest
// frame is dropped when a new one would exceed it.  Frame `k` is the state after `k * snapEvery` steps, so the frame
// index IS the step count.
//
// A frame can only be DISPLAYED from the window; to RESUME the simulation from one the law is run again, which is
// exact because the law is deterministic:
//   * KEYFRAMES are full copies of the law's state (opaque to this module; the runtime makes and frees them) taken
//     every `keyEvery` steps, thinned when they pass `keyBytes`;
//   * the INPUT LOG is every runtime law input (gravity, ...) with the step it was set at;
//   * `plan(frame)` is the exact recipe -- the keyframe to restore, then the steps to re-run with each logged input
//     set at its step -- that reproduces the state of that frame bit for bit.
export class Recording {
  constructor({ snapEvery, frameBytes, capBytes, keyBytes = capBytes / 4, keyEvery = snapEvery * 60 }) {
    this.snapEvery = snapEvery; this.frameBytes = frameBytes; this.keyEvery = keyEvery; this.keyBytes = keyBytes;
    this.capFrames = Math.max(2, Math.floor(capBytes / frameBytes));
    this.clear();
  }

  clear() {
    const old = this.frames ? this.frames.slice(this.head).filter(Boolean) : [];
    this.frames = []; this.head = 0; this.lo = 1; this.hi = 0;          // an empty window: lo > hi
    const keys = this.keys || [];
    this.keys = []; this.inputs = [];
    return { frames: old, keys };
  }

  get length() { return this.hi - this.lo + 1; }
  get bytes() { return this.length * this.frameBytes; }
  get keyBytesUsed() { return this.keys.reduce((s, k) => s + (k.bytes || 0), 0); }
  stepOf(frame) { return frame * this.snapEvery; }
  frameOf(step) { return Math.floor(step / this.snapEvery); }
  get(k) { return k < this.lo || k > this.hi ? null : this.frames[this.head + k - this.lo]; }

  // Append frame `k` (must be hi + 1).  Returns the frames and keyframes that fell out of the window, to be recycled.
  push(k, data) {
    if (k !== this.hi + 1) throw new Error(`recording: frame ${k} is not next after ${this.hi}`);
    if (this.hi < this.lo) this.lo = k;
    this.frames.push(data); this.hi = k;
    const out = { frames: [], keys: [] };
    while (this.length > this.capFrames) { out.frames.push(this.frames[this.head]); this.frames[this.head++] = undefined; this.lo++; }
    if (this.head > 1024 && this.head * 2 > this.frames.length) { this.frames = this.frames.slice(this.head); this.head = 0; }
    out.keys = this.#prune();
    return out;
  }

  // Drop everything after frame `f` (the run is about to branch from it).
  truncateAfter(f) {
    const out = { frames: [], keys: [] };
    while (this.hi > f && this.hi >= this.lo) { out.frames.push(this.frames.pop()); this.hi--; }
    const step = this.stepOf(f);
    out.keys = this.keys.filter((k) => k.step > step);
    this.keys = this.keys.filter((k) => k.step <= step);
    this.inputs = this.inputs.filter((e) => e.step <= step);
    return out;
  }

  // ---- keyframes
  addKey(step, snap, bytes = 0) {
    this.keys.push({ step, snap, bytes });
    const dropped = this.#prune();
    while (this.keyBytesUsed > this.keyBytes && this.keys.length > 2) {
      // thin: drop every second key (never the oldest or the newest)
      const keep = [], thin = [];
      this.keys.forEach((k, i) => (i === 0 || i === this.keys.length - 1 || i % 2 === 0 ? keep : thin).push(k));
      this.keys = keep; dropped.push(...thin);
    }
    return dropped;
  }

  // keys older than the last one at or before the window's first frame are of no use to any frame in the window
  #prune() {
    if (this.hi < this.lo) return [];
    const need = this.stepOf(this.lo);
    let last = -1;
    this.keys.forEach((k, i) => { if (k.step <= need) last = i; });
    const dropped = last > 0 ? this.keys.splice(0, last) : [];
    const base = this.keys.length ? this.keys[0].step : 0;
    if (this.keys.length) {
      let at = -1;
      this.inputs.forEach((e, i) => { if (e.step <= base) at = i; });
      if (at > 0) this.inputs.splice(0, at);
    }
    return dropped;
  }

  // ---- the input log
  // Set the runtime inputs to `values` (an object) as of `step`: they apply to every step after it.
  setInputs(step, values) {
    const last = this.inputs[this.inputs.length - 1];
    const same = (a, b) => Object.keys(a).length === Object.keys(b).length && Object.keys(a).every((n) => a[n] === b[n]);
    if (last && last.step === step) last.values = { ...values };
    else if (!last || !same(last.values, values)) this.inputs.push({ step, values: { ...values } });
  }

  inputsAt(step) {
    let found = null;
    for (const e of this.inputs) { if (e.step <= step) found = e; else break; }
    return found ? found.values : null;
  }

  // The recipe for the state of frame `f`: restore `key` (null: the start state), set `initial`, then run `ops`
  // (each `{steps}` or `{set}`, in order).
  plan(f) {
    const target = this.stepOf(f);
    let key = null;
    for (const k of this.keys) if (k.step <= target) key = k;
    const start = key ? key.step : 0, ops = [];
    let at = start;
    for (const e of this.inputs) {
      if (e.step <= start || e.step >= target) continue;
      ops.push({ steps: e.step - at }, { set: e.values }); at = e.step;
    }
    ops.push({ steps: target - at });
    return { key, start, target, initial: this.inputsAt(start), ops: ops.filter((o) => o.set || o.steps > 0) };
  }
}

// The display cursor.  `mode`: "live" follows the newest frame at the pace the caller gives; "replay" plays the window
// from its first frame to its last and holds; "bounce" plays it forward then backward, for ever.
export class Playback {
  constructor() { this.mode = "live"; this.pos = 0; this.dir = 1; this.ended = false; }

  set(mode, rec) {
    this.mode = mode; this.dir = 1; this.ended = false;
    if (mode === "replay" || mode === "bounce") this.pos = rec.lo;
    else if (mode === "live") this.pos = Math.max(rec.lo, Math.min(this.pos, rec.hi));
  }

  jump(frame) { this.pos = frame; }

  // Move the cursor `frames` frames; returns the frame to draw (or null when the window is empty).
  advance(frames, rec, { turbo = false, maxLead = Infinity } = {}) {
    const { lo, hi } = rec;
    if (hi < lo) return null;
    if (this.mode === "live") {
      this.pos = turbo ? hi : Math.min(this.pos + frames, hi);
      if (hi - this.pos > maxLead) this.pos = hi - maxLead;
    } else if (this.mode === "replay") {
      if (!this.ended) { this.pos += frames; if (this.pos >= hi) { this.pos = hi; this.ended = true; } }
    } else if (hi > lo) {
      this.pos += this.dir * frames;
      for (let guard = 0; guard < 8 && (this.pos > hi || this.pos < lo); guard++) {
        if (this.pos > hi) { this.pos = 2 * hi - this.pos; this.dir = -1; }
        if (this.pos < lo) { this.pos = 2 * lo - this.pos; this.dir = 1; }
      }
    } else this.pos = lo;
    this.pos = Math.max(lo, Math.min(this.pos, hi));
    return Math.floor(this.pos);
  }

  lead(rec) { return rec.hi - Math.floor(this.pos); }
}
