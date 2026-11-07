// Node check of web/replay_layers.js: layer sync (step mapping, loop / hold / bounce), the identity join, the divergence metric,
// the memory cap, the reduced frame, the plane plan, settings round trips and a ghost run against a fake runtime.
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../examples/meowdoku/web");
const R = await import(pathToFileURL(path.join(web, "replay_layers.js")).href);
const { Recording } = await import(pathToFileURL(path.join(web, "recording.js")).href);
const near = (a, b, tol = 1e-9, m = "") => assert.ok(Math.abs(a - b) <= tol, `${m} ${a} != ${b}`);

// a packed frame: grain i at (i * 0.01 + t, 0.5 + 0.002 i, 0) in the live format [x, y, z, rgba bits, meta]
function frame(n, t = 0, dead = []) {
  const f = new Float32Array(5 * n), bits = new Uint32Array(f.buffer);
  for (let i = 0; i < n; i++) {
    f[5 * i] = i * 0.01 + t; f[5 * i + 1] = 0.5 + 0.002 * i; f[5 * i + 2] = 0;
    bits[5 * i + 3] = ((255 << 24) | ((i % 256) << 16) | (40 << 8) | 200) >>> 0;
    f[5 * i + 4] = dead.includes(i) ? 0 : (i % 2 ? 100 : 200) + 12;                      // flag * 100 + radius * 1000
  }
  return f;
}

// ---- step mapping: the live run's displayed frame -> the layer's frame
{
  const L = { first: 1, count: 5 };
  assert.equal(R.stepMap(0, { ...L, mode: "loop" }), -1, "before the layer starts");
  assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 11].map((s) => R.stepMap(s, { ...L, mode: "loop" })), [0, 1, 2, 3, 4, 0, 1, 0]);
  assert.deepEqual([1, 5, 6, 7, 99].map((s) => R.stepMap(s, { ...L, mode: "hold" })), [0, 4, 4, 4, 4]);
  assert.deepEqual([1, 2, 5, 6, 7, 8, 9, 10].map((s) => R.stepMap(s, { ...L, mode: "bounce" })), [0, 1, 4, 3, 2, 1, 0, 1]);
  assert.equal(R.stepMap(3, { first: 1, count: 0, mode: "loop" }), -1, "an empty layer");
  assert.equal(R.stepMap(null, L), -1);
  assert.equal(R.stepMap(7, { first: 3, count: 1, mode: "bounce" }), 0);
  assert.equal(R.stepMap(9, { first: 4, count: 3, mode: "loop" }), 2, "a layer that starts at frame 4");
  // two layers of different length share one clock: same step, each its own frame
  const a = { first: 1, count: 4, mode: "loop" }, b = { first: 1, count: 6, mode: "hold" };
  assert.deepEqual([5, 6, 7].map((s) => [R.stepMap(s, a), R.stepMap(s, b)]), [[0, 4], [1, 5], [2, 5]]);
}

// ---- identity: global ids and the join on the grain id
{
  assert.equal(R.gidLayer(R.gid(3, 12345)), 3);
  assert.equal(R.gidId(R.gid(3, 12345)), 12345);
  assert.notEqual(R.gid(1, 7), R.gid(2, 7), "the same grain id in two layers is two identities");
  const [a, b] = R.joinIds(R.rowIds(6), R.rowIds(6));
  assert.deepEqual([...a], [0, 1, 2, 3, 4, 5]); assert.deepEqual([...b], [...a]);
}
{
  const [ra, rb] = R.joinIds(Uint32Array.of(0, 2, 4, 6), Uint32Array.of(0, 1, 2, 3, 4, 5));
  assert.deepEqual([...ra], [0, 1, 2]); assert.deepEqual([...rb], [0, 2, 4]); // ids 0, 2, 4 are in both; 6 is only in a
  const [sa, sb] = R.joinIds(Uint32Array.of(5, 3, 9), Uint32Array.of(9, 5));
  assert.deepEqual([[...sa], [...sb]], [[0, 2], [1, 0]], "unsorted ids join by value");
}

// ---- the reduced frame
{
  const n = 64, f = frame(n, 0, [5, 6]), r = R.reduceFrame(f, n);
  assert.equal(R.reducedBytes(n), 8 * n + 24);
  assert.equal(r.q.byteLength + r.bounds.byteLength, R.reducedBytes(n), "8 bytes per grain plus the bounds");
  assert.ok(R.reducedBytes(n) < R.fullBytes(n) * 0.5, "less than half of the full frame");
  const p = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (i === 5 || i === 6) { assert.equal(r.q[4 * i + 3], 0, "a dead grain stays dead"); continue; }
    R.positionOf(r, i, p);
    near(p[0], f[5 * i], 2e-5, "x"); near(p[1], f[5 * i + 1], 2e-5, "y");
    near(R.metaOf(r, i), f[5 * i + 4], 1 / 64 + 1e-9, "meta (flag and radius)");
  }
  // an all-dead frame reduces without NaN
  const e = R.reduceFrame(new Float32Array(5 * 4), 4);
  assert.deepEqual([...e.bounds], [0, 0, 0, 0, 0, 0]);
}

// ---- the divergence metric
{
  near(R.divergenceOf(0, 0, 0, 3, 4, 0), 5);
  near(R.divergenceOf(1, 1, 1, 1, 1, 1), 0);
  assert.equal(R.divergenceUnit(0.5, 0.25), 1, "saturates at the scale");
  near(R.divergenceUnit(0.1, 0.4), 0.25);
  assert.deepEqual(R.divergenceRamp(0), [40, 90, 255]); assert.deepEqual(R.divergenceRamp(1), [255, 50, 40]);
  // the ghost frame: grain i of the layer is the live grain i moved by 0.03 * i in x -> divergence 0.03 * i
  const n = 20, live = frame(n, 0), ghost = frame(n, 0);
  for (let i = 0; i < n; i++) ghost[5 * i] += 0.03 * i;
  const out = new Float32Array(5 * n), st = R.buildGhostFrame(out, R.reduceFrame(ghost, n), live, { diverge: { scale: 0.3 }, alpha: 0.5 });
  assert.equal(st.joined, n);
  near(st.mean, 0.03 * (n - 1) / 2, 2e-4, "mean divergence"); near(st.max, 0.03 * (n - 1), 2e-4, "max divergence");
  const bits = new Uint32Array(out.buffer);
  const rgb = (i) => [bits[5 * i + 3] & 255, (bits[5 * i + 3] >> 8) & 255, (bits[5 * i + 3] >> 16) & 255, bits[5 * i + 3] >>> 24];
  assert.deepEqual(rgb(0).slice(0, 3), R.divergenceRamp(0), "together: the cool end");
  assert.deepEqual(rgb(n - 1).slice(0, 3), R.divergenceRamp(R.divergenceUnit(0.03 * (n - 1), 0.3)), "far apart: by the ramp");
  assert.equal(rgb(3)[3], 128, "alpha is the layer's opacity");
  // identical copies do not diverge at all
  const same = R.buildGhostFrame(new Float32Array(5 * n), R.reduceFrame(live, n), live, { diverge: { scale: 0.3 } });
  assert.ok(same.max < 2e-5 && same.over === 0);
}

// ---- tint and the live colour: the ghost wears the live grain's colour pulled to the tint; a grain dead in the live frame has no divergence
{
  const n = 8, live = frame(n, 0, [2]), ghost = R.reduceFrame(frame(n, 0.01), n);
  const out = new Float32Array(5 * n), st = R.buildGhostFrame(out, ghost, live, { tint: [0, 255, 0], tintAmount: 1, alpha: 1, diverge: { scale: 1 } });
  assert.equal(st.joined, n - 1, "grain 2 is dead in the live frame: not joined");
  const o2 = new Float32Array(5 * n); R.buildGhostFrame(o2, ghost, live, { tint: [0, 255, 0], tintAmount: 1, alpha: 1 });
  const c = new Uint32Array(o2.buffer)[3]; assert.deepEqual([c & 255, (c >> 8) & 255, (c >> 16) & 255], [0, 255, 0]);
  const o3 = new Float32Array(5 * n); R.buildGhostFrame(o3, ghost, live, { tintAmount: 0, alpha: 1 });
  assert.equal(new Uint32Array(o3.buffer)[3] & 255, 200, "no tint: the live grain's own colour");
  // a layer of another stride joins on the id
  const big = frame(16, 0), small = R.reduceFrame(frame(8, 0), 8);          // the layer shows every 2nd id of the live run (rows 0..7 = ids 0, 2, ..., 14)
  for (let i = 0; i < 8; i++) { /* live row 2i is the layer's row i */ }
  const o4 = new Float32Array(5 * 8), s4 = R.buildGhostFrame(o4, small, big, { diverge: { scale: 1 }, stride: 2, liveStride: 1 });
  assert.equal(s4.joined, 8);
}

// ---- the memory cap
{
  const n = 1000, per = R.reducedBytes(n), mk = (frames) => new R.ReplayLayer({ id: 0, frames: Array.from({ length: frames }, () => R.reduceFrame(frame(n), n)), first: 1 });
  const set = new R.LayerSet(per * 100);
  assert.equal(set.add(mk(40)), 0); assert.equal(set.add(mk(40)), 1);
  assert.equal(set.add(mk(40)), -1, "a third layer would pass the cap");
  assert.equal(set.used, 2); assert.equal(set.bytes, per * 80); assert.equal(set.room, per * 20);
  assert.equal(set.add(mk(20)), 2, "exactly the room left fits");
  assert.equal(set.add(mk(1)), -1);
  assert.equal(set.slots[1].id, 2, "layer ids count from 1; 0 is the live run");
  set.remove(0); assert.equal(set.room, per * 40);
  assert.equal(set.add(mk(10), 1), 1, "replacing a slot counts the layer it replaces"); assert.equal(set.bytes, per * 30);
  assert.equal(new R.LayerSet(per * 100, 1).add(mk(1), 3), -1, "no such slot");
  // the plan: how many frames a ghost can have so that K fit
  const p = R.planGhosts({ room: per * 90, k: 3, n, snapEvery: 6, dt: 1 / 360, seconds: 4 });
  assert.equal(p.wanted, 240); assert.equal(p.frames, 30); assert.ok(p.truncated); assert.equal(p.total, p.frames * per * 3); assert.ok(p.total <= per * 90);
  const q = R.planGhosts({ room: per * 9000, k: 3, n, snapEvery: 6, dt: 1 / 360, seconds: 4 });
  assert.equal(q.frames, 240); assert.ok(!q.truncated);
  assert.equal(R.planGhosts({ room: 10, k: 2, n, snapEvery: 6, dt: 1 / 360, seconds: 4 }).frames, 0);
  // the layer taken from the live recorder keeps the newest frames that fit and reads the window (not changing it)
  const tape = new Recording({ snapEvery: 6, frameBytes: n * 20, capBytes: n * 20 * 200 });
  for (let k = 1; k <= 50; k++) tape.push(k, frame(n, k * 0.001));
  const lay = R.layerFromRecording(tape, { room: per * 30 });
  assert.equal(lay.count, 30); assert.equal(lay.first, 21); assert.equal(lay.bytes, per * 30);
  assert.equal(tape.length, 50, "the recorder is untouched"); assert.equal(R.layerFromRecording(tape, { room: per }), null);
  assert.equal(R.stepMap(21, lay), 0, "the layer's first frame is the run's frame 21");
  near(R.positionOf(lay.frameAt(21), 0)[0], 21 * 0.001, 2e-5);
}

// ---- planning the draw: planes behind each other, or one layer; disabled and unstarted layers are skipped
{
  const n = 10, set = new R.LayerSet(1e9), s = R.sanitizeSettings(null);
  const mk = () => new R.ReplayLayer({ id: 0, frames: [R.reduceFrame(frame(n), n), R.reduceFrame(frame(n, 1), n)], first: 1 });
  set.add(mk()); set.add(mk()); set.add(mk());
  assert.deepEqual(R.planDraw(set, s, 1, 4), [], "comp off");
  s.comp = "behind"; s.depth = 0.1;
  let plan = R.planDraw(set, s, 2, 4);
  assert.deepEqual(plan.map((p) => p.slot), [2, 1, 0], "the farthest plane is drawn first");
  assert.deepEqual(plan.map((p) => p.zOffset.toFixed(2)), ["-1.20", "-0.80", "-0.40"], "planes depth * width apart behind the live one");
  assert.ok(plan.every((p) => p.frame === p.layer.frames[1]));
  s.slots[1].on = false;
  assert.deepEqual(R.planDraw(set, s, 2, 4).map((p) => [p.slot, p.zOffset.toFixed(2)]), [[2, "-0.80"], [0, "-0.40"]], "a switched-off layer leaves no gap");
  s.comp = "single";
  assert.ok(R.planDraw(set, s, 2, 4).every((p) => p.zOffset === 0), "composed: all in the live plane");
  assert.deepEqual(R.planDraw(set, s, 0, 4), [], "before the layers start");
  s.slots[0].alpha = 0; assert.ok(!R.planDraw(set, s, 2, 4).some((p) => p.slot === 0), "opacity 0 draws nothing");
}

// ---- settings: the share link carries them (applySetting / settingValue), clamped
{
  const s = R.DEFAULT_SETTINGS();
  R.applySetting(s, "ghosts_comp", "behind"); R.applySetting(s, "ghosts2_alpha", 0.3); R.applySetting(s, "ghosts2_tint", "#AABBCC"); R.applySetting(s, "ghosts3_on", false);
  R.applySetting(s, "ghosts_depth", 9); R.applySetting(s, "ghosts_k", 99); R.applySetting(s, "ghosts_comp", "nonsense");
  assert.equal(s.comp, "off"); assert.equal(s.depth, 0.5); assert.equal(s.k, R.MAX_LAYERS);
  assert.equal(R.settingValue(s, "ghosts2_alpha"), 0.3); assert.equal(R.settingValue(s, "ghosts2_tint"), "#aabbcc"); assert.equal(R.settingValue(s, "ghosts3_on"), false);
  const again = R.sanitizeSettings(JSON.parse(JSON.stringify(s)));
  assert.deepEqual(again, s, "a settings round trip is exact");
  assert.deepEqual(R.sanitizeSettings({ comp: 7, slots: [null, { alpha: "x" }] }), R.DEFAULT_SETTINGS(), "garbage falls back to the defaults");
}

// ---- the perturbation: seeded, deterministic, seed 0 is the control
{
  assert.deepEqual(R.perturbation(0), { dtheta: 0, dnozzle: 0 });
  assert.deepEqual(R.perturbation(5, 1), R.perturbation(5, 1)); assert.notDeepEqual(R.perturbation(5, 1), R.perturbation(6, 1));
  near(R.perturbation(5, 2).dtheta, 2 * R.perturbation(5, 1).dtheta, 1e-12, "linear in the amplitude");
  assert.deepEqual(R.perturbation(5, 0), { dtheta: 0, dnozzle: 0 });
}

// ---- saving a layer
{
  const n = 12, lay = new R.ReplayLayer({ id: 0, name: "x", seed: 3, first: 7, stride: 2, frames: [R.reduceFrame(frame(n), n), R.reduceFrame(frame(n, 0.5), n)] });
  const back = R.deserializeLayer(R.serializeLayer(lay));
  assert.equal(back.count, 2); assert.equal(back.first, 7); assert.equal(back.stride, 2); assert.equal(back.seed, 3);
  assert.deepEqual([...back.frames[1].q], [...lay.frames[1].q]); assert.deepEqual([...back.frames[0].bounds], [...lay.frames[0].bounds]);
}

// ---- making ghosts: a fake runtime that records frames the way ComputeRuntime does (checks the recorder is used as it is)
{
  const n = 30, runs = [];
  class Fake {
    constructor() { this.ready = false; this.error = null; this.inflight = 0; this.running = true; this.tape = null; runs.push(this); this.disposed = false; }
    setRecording(on) { this.rec = on; }
    inheritFrom(from, o) { this.inherit = o; }
    perturbedUniforms(u, rel) { return { ...u, nozzle_speed: u.nozzle_speed * (1 + rel) }; }
    sync(spec) {
      this.spec = spec;
      setTimeout(() => {
        this.tape = new Recording({ snapEvery: 6, frameBytes: n * 20, capBytes: this.capBytes }); this.ready = true;
        this.timer = setInterval(() => { if (!this.running || this.disposed) return; const k = this.tape.hi + 1; this.tape.push(k, frame(n, k * 0.001 * (1 + this.inherit.nozzleScale - 1))); }, 1);
      }, 2);
    }
    dispose() { this.disposed = true; clearInterval(this.timer); }
  }
  const live = { tape: { frameBytes: n * 20 }, spec: { speed: 1, uniforms: { nozzle_speed: 2 } } };
  const progress = [], got = [];
  const layers = await R.makeGhosts({ live, makeRuntime: () => new Fake(), seeds: [0, 4, 9], frames: 12, amp: 1, stride: 1, onProgress: (p) => progress.push(p), onLayer: (l, k) => got.push([k, l.seed]), sleepMs: 1 });
  assert.equal(layers.length, 3); assert.deepEqual(got, [[0, 0], [1, 4], [2, 9]]);
  assert.ok(layers.every((l) => l.count === 12 && l.first === 1 && l.n === n), "12 frames from frame 1 each");
  assert.ok(runs.every((r) => r.disposed), "every ghost runtime is freed");
  assert.equal(runs[0].spec.speed, 0, "ghost runs go flat out"); assert.equal(runs[0].spec.uniforms.nozzle_speed, 2, "seed 0: unperturbed");
  assert.notEqual(runs[1].spec.uniforms.nozzle_speed, 2, "a seed perturbs the nozzle speed");
  assert.equal(runs[2].inherit.caseTiltAdd, R.perturbation(9, 1).dtheta);
  assert.ok(progress.some((p) => p.ghost === 3));
  // stopping between ghosts keeps the finished ones
  let stop = false;
  const some = await R.makeGhosts({ live, makeRuntime: () => new Fake(), seeds: [1, 2, 3], frames: 6, onLayer: () => { stop = true; }, shouldStop: () => stop, sleepMs: 1 });
  assert.equal(some.length, 1);
}

console.log("replay layers: ok");
