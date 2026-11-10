// Actual ComputeRuntime state/replay paths with an in-memory product double.
// This tests handoff bookkeeping, not GPU execution or new collision physics.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {fileURLToPath, pathToFileURL} from "node:url";
import {spawnSync} from "node:child_process";
import {decodeStateStart, decodeStateOrientations, encodeStateStart} from "../examples/meowdoku/web/compute_state.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const web = path.join(root, "examples/meowdoku/web");
const temp = await fs.mkdtemp(path.join(os.tmpdir(), "meowdoku-state-"));
const messageChannel = globalThis.MessageChannel;
globalThis.MessageChannel = undefined; // do not run a pump against the test product
let checks = 0;
const check = async (name, fn) => { await fn(); checks++; console.log(`PASS ${name}`); };
try {
  // Match server.py's existing shell publication, in an isolated test folder.
  for (const name of ["compute.js", "compute_state.js", "case_vibration.js", "recording.js", "particle_styles.js",
    "world_curve.js", "world_zoom.js", "noise_filter.js", "mobroad.js", "case_motion.js", "marble_rotation.js", "emoji_tumble.js", "marble_sim_preview.js"])
    await fs.copyFile(path.join(web, name), path.join(temp, name));
  await fs.copyFile(path.join(root, "examples/mobile_reference_app/web/turing-wasm-shell.js"), path.join(temp, "turing-wasm-shell.js"));
  await fs.writeFile(path.join(temp, "package.json"), '{"type":"module"}');
  // Export unchanged product classes only in the test copy for readback checks.
  await fs.appendFile(path.join(temp, "compute.js"), "\nexport { GpuProduct, WasmProduct };\n");
  const {ComputeRuntime, decodeStart, GpuProduct, WasmProduct} = await import(pathToFileURL(path.join(temp, "compute.js")));
  const {Recording} = await import(pathToFileURL(path.join(temp, "recording.js")));
  const {integrateAngular} = await import(pathToFileURL(path.join(temp, "marble_rotation.js")));
  const names = ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "age", "rs", "bth", "bom"];
  const pad = 4, n = 2, snap = 2, dt = .125;
  const initial = Object.fromEntries(names.map((name, k) => [name, Float32Array.from([k + .125, k + .25, k + .5, -0])]));
  initial.age.set([2, 3, -1e9, -1e9]); initial.rs.set([.01, .02, .03, .04]);
  const q0 = [0, Math.SQRT1_2, 0, Math.SQRT1_2], q1 = [1, 0, 0, 0];
  const orientations = [q0, q1];
  const encoded = encodeStateStart(initial, pad, names, orientations);

  await check("existing simulation preview suite passes with published shell dependency", async () => {
    const source = await fs.readFile(path.join(root, "tests/meowdoku_marble_sim_preview_check.mjs"), "utf8");
    const target = path.join(temp, "legacy-preview.mjs");
    await fs.writeFile(target, source.replaceAll("../examples/meowdoku/web/", "./"));
    const result = spawnSync(process.execPath, [target], {encoding: "utf8", timeout: 10000});
    assert.equal(result.status, 0, result.stdout + result.stderr);
  });

  await check("all native columns, inactive padding and signed zero roundtrip", () => {
    const decoded = decodeStart(encoded, pad, names);
    for (const name of names) assert.deepEqual(new Uint8Array(decoded[name].buffer), new Uint8Array(initial[name].buffer));
    assert.deepEqual(decodeStateOrientations(encoded, 2, decoded).map(row => row.quaternion), orientations);
    assert.notEqual(encoded.orientations_xyzw[0], q0);
    const partial = Object.fromEntries(names.map(name => [name, initial[name].slice(0, 2)]));
    assert.deepEqual(Array.from(decodeStateStart(encodeStateStart(partial, pad, names), pad, names).age), [2, 3, 0, 0]);
  });

  await check("strict manifest names, lengths, finite float32 and format rejection", () => {
    assert.throws(() => decodeStart(encoded, pad), /manifest/);
    assert.throws(() => decodeStart({...encoded, age_f32: "AAAAAA=="}, pad, names), /legacy/);
    for (const packed of [{...encoded.state_f32, unknown: ""}, {...encoded.state_f32, vx: "AA=="}, {...encoded.state_f32, vx: 1}])
      assert.throws(() => decodeStateStart({state_f32: packed}, pad, names));
    const missing = {...encoded.state_f32}; delete missing.wz;
    assert.throws(() => decodeStateStart({state_f32: missing}, pad, names), /exactly match/);
    for (const value of [NaN, Infinity, -Infinity, 1e100, "2"])
      assert.throws(() => encodeStateStart({...initial, vx: [value, 1, 2, 3]}, pad, names), /float32/);
    assert.throws(() => encodeStateStart({...initial, vx: [1]}, pad, names), /different lengths/);
    const bad = Buffer.alloc(pad * 4); bad.writeFloatLE(NaN, 0);
    assert.throws(() => decodeStateStart({state_f32: {...encoded.state_f32, vx: bad.toString("base64")}}, pad, names), /nonfinite/);
    assert.throws(() => decodeStateStart(encoded, pad, [...names, "x"]), /column names/);
    assert.throws(() => decodeStateOrientations({...encoded, orientations_xyzw: [[0,0,0,0], q1]}, 2, initial), /unit/);
    assert.throws(() => decodeStateOrientations(encoded, 1, initial), /count/);
    assert.throws(() => decodeStateOrientations(encoded, 2, {wx: initial.wx}), /count/);
  });

  await check("legacy age and fixed-point starts retain old decode", () => {
    assert.deepEqual(decodeStart({age_f32: Buffer.from(initial.age.buffer).toString("base64")}, pad).age, initial.age);
    const xyz = Buffer.alloc(12); [0, 65535, 32768, 65535, 0, 0].forEach((v, i) => xyz.writeUInt16LE(v, i * 2));
    const legacy = decodeStart({xyz: xyz.toString("base64"), n: 2, bounds: [-1, 1, 2, 4, 0, 10]}, pad);
    assert.deepEqual(Array.from(legacy.x), [-1, 1, 0, 0]);
    assert.deepEqual(Array.from(legacy.y), [4, 2, 0, 0]);
  });

  class Product {
    constructor() { this.pool = []; this.frameBytes = 40; this.resets = 0; this.restores = 0; this.stepped = 0; this.reset({named: initial, uniforms: {force: 1}}); }
    reset({named, uniforms}) {this.resets++; this.state = Object.fromEntries(names.map(name => [name, new Float32Array(named[name] || pad)])); this.uniforms = {...uniforms};}
    setUniform(name, value) {this.uniforms[name] = value;}
    step(count) {
      this.stepped += count;
      for (let s = 0; s < count; s++) for (let i = 0; i < pad; i++) {
        this.state.vx[i] += this.uniforms.force * dt;
        for (const [p, v] of [["x","vx"],["y","vy"],["z","vz"]]) this.state[p][i] += this.state[v][i] * dt;
        this.state.age[i] += dt;
      }
    }
    restore(state) {this.restores++; this.state = Object.fromEntries(names.map(name => [name, new Float32Array(state[name])]));}
    freeKey() {}
    async idle() {}
    async columnsNow(requested, count = n) {this.readCount = count; return Object.fromEntries(requested.map(name => [name, this.state[name].slice(0, count)]));}
  }
  function fixture({key = false, end = 5, shown = 3, tracked = 2} = {}) {
    const runtime = new ComputeRuntime({onStatus() {}, onSamples() {}});
    runtime.manifestName = "fixture.manifest.json";
    runtime.manifest = {law: {state: names, dt, runtime_inputs: ["force"], gravity: 1},
      points: {x: "x", y: "y", z: "z"},
      settle: {sample: 2, max_seconds: 40}, sizes: {small: {n, pad, display_stride: 1, uniforms: {force: 1}}}};
    runtime.spec = {manifest: runtime.manifestName, size: "small", epoch: 1, uniforms: {force: 1}, studio_count: tracked};
    runtime.product = new Product(); runtime.productKey = `${runtime.manifestName}:small:angular${tracked}`;
    runtime.support = {small: {ok: true}};
    runtime.ready = true; runtime.running = true; runtime.epoch = 1; runtime.uniforms = {force: 1};
    runtime.start = Object.fromEntries(names.map(name => [name, initial[name].slice()]));
    runtime.tape = new Recording({snapEvery: snap, frameBytes: 40, capBytes: 1e6});
    runtime.tape.setInputs(0, {force: 1});
    const expected = new Map();
    for (let f = 1; f <= end; f++) {
      if (f === 2) {runtime.tape.setInputs(snap, {force: 2}); runtime.product.setUniform("force", 2);}
      runtime.product.step(snap);
      const frame = new Float32Array(10);
      for (let id = 0; id < n; id++) for (const [component, name] of ["x", "y", "z"].entries())
        frame[id * 5 + component] = runtime.product.state[name][id];
      runtime.tape.push(f, frame);
      expected.set(f, Object.fromEntries(names.map(name => [name, runtime.product.state[name].slice()])));
      if (f === 2 && key) runtime.tape.addKey(f * snap, expected.get(f), 208);
      runtime.rotations.set(f, orientations.slice(0, tracked).map((q, id) => ({frame: f,
        quaternion: integrateAngular(q, [1,2,3], f * snap * dt), omega: [initial.wx[id], initial.wy[id], initial.wz[id]]})));
    }
    runtime.steps = end * snap; runtime.shown = shown; runtime.play.jump(shown);
    return {runtime, expected};
  }

  await check("handoff reconstructs shown frame using existing input log, not queued future", async () => {
    const {runtime, expected} = fixture();
    const result = await runtime.pauseAtDisplayedState();
    const decoded = decodeStart(result.start, pad, names);
    for (const name of names) assert.deepEqual(decoded[name], expected.get(3)[name], name);
    assert.equal(runtime.product.readCount, pad);
    assert.equal(result.step, 6); assert.equal(result.seconds, .75); assert.equal(result.uniforms.force, 2);
    assert.deepEqual(result.clock,{dt:.125,snapEvery:2,epoch:1});
    assert.equal(result.product.count,n);assert.equal(result.product.pad,pad);
    assert.deepEqual(result.start.orientations_xyzw, runtime.rotations.get(3).map(row => row.quaternion));
    assert.equal(result.caseAxis, "y"); assert.equal(result.caseAngle, 0);
    assert.equal(runtime.steps, 6); assert.equal(runtime.tape.hi, 3); assert.equal(runtime.running, false);
    assert.equal(runtime.rotations.has(4), false); assert.equal(runtime.handoffPaused, true);
    const second = await runtime.pauseAtDisplayedState({maxReplaySteps: 0});
    assert.deepEqual(second, result);
    runtime.frame(1000); assert.equal(runtime.shown, 3); assert.equal(runtime.play.pos, 3);
    const resets = runtime.product.resets;
    runtime.control("continue"); await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(runtime.handoffPaused, false); assert.equal(runtime.running, true);
    assert.equal(runtime.steps, 6); assert.equal(runtime.product.resets, resets);
  });

  await check("keyframe handoff uses original restore and exact remaining steps", async () => {
    const {runtime, expected} = fixture({key: true});
    const stepped = runtime.product.stepped, resets = runtime.product.resets;
    const result = await runtime.pauseAtDisplayedState({maxReplaySteps: 2});
    assert.equal(runtime.product.restores, 1); assert.equal(runtime.product.resets, resets);
    assert.equal(runtime.product.stepped - stepped, 2);
    for (const name of names) assert.deepEqual(decodeStart(result.start, pad, names)[name], expected.get(3)[name]);
  });

  await check("latest-frame handoff reads without resetting or replaying", async () => {
    const {runtime} = fixture({shown: 5}); const resets = runtime.product.resets, stepped = runtime.product.stepped;
    await runtime.pauseAtDisplayedState({maxReplaySteps: 0});
    assert.equal(runtime.product.resets, resets); assert.equal(runtime.product.stepped, stepped);
  });

  await check("pending batches settle before readback and repeated requests coalesce", async () => {
    const {runtime} = fixture(); runtime.inflight = 1;
    const promise = runtime.pauseAtDisplayedState(); assert.equal(runtime.pauseAtDisplayedState(), promise);
    await new Promise(resolve => setTimeout(resolve, 5)); assert.equal(runtime.product.readCount, undefined);
    runtime.inflight = 0; const result = await promise; assert.equal(result.frame, 3);
  });

  await check("equivalent spec refresh keeps the authoritative run identity", async () => {
    const {runtime} = fixture(); runtime.inflight = 1;
    const promise = runtime.pauseAtDisplayedState();
    runtime.sync({...runtime.spec, uniforms: {...runtime.spec.uniforms}});
    runtime.inflight = 0;
    const result = await promise; await runtime.queue;
    assert.equal(result.frame, 3); assert.equal(runtime.error, null);
    assert.equal(runtime.ready, true); assert.equal(runtime.handoffPaused, true);
    runtime.control("continue"); await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(runtime.running, true); assert.equal(runtime.handoffPaused, false);
  });

  await check("Replay and Bounce leave handoff pause and advance the recording", async () => {
    for (const mode of ["replay", "bounce"]) for (const failed of [false, true]) {
      const {runtime} = fixture();
      if (failed) {runtime.rotations.delete(3); await assert.rejects(runtime.pauseAtDisplayedState(), /orientations/);}
      else await runtime.pauseAtDisplayedState();
      runtime.control(mode); runtime.last = 0;
      assert.equal(runtime.handoffPaused, false); assert.equal(runtime.play.mode, mode); assert.equal(runtime.shown, 1);
      runtime.frame(.1); runtime.frame(.2); runtime.frame(.3);
      assert.equal(runtime.shown, 2); assert.equal(runtime.running, false);
      assert.equal(runtime.resimming, false);
    }
  });

  await check("playback can interrupt a pending handoff without a stale result", async () => {
    for (const mode of ["replay", "bounce", "continue"]) {
      const {runtime} = fixture(); runtime.inflight = 1;
      const promise = runtime.pauseAtDisplayedState();
      runtime.control(mode); runtime.inflight = 0;
      await assert.rejects(promise, /interrupted by playback/);
      await new Promise(resolve => setTimeout(resolve, 0));
      assert.equal(runtime.handoffPaused, false); assert.equal(runtime.resimming, false);
      assert.equal(runtime.play.mode, mode === "continue" ? "live" : mode);
      assert.equal(runtime.running, mode === "continue");
      assert.equal(runtime.product.readCount, undefined);
    }
  });

  await check("playback cancellation also waits for an already-started column readback", async () => {
    for (const mode of ["replay", "bounce", "continue"]) {
      const {runtime} = fixture();
      const read = runtime.product.columnsNow.bind(runtime.product);
      let release, entered = false;
      const gate = new Promise(resolve => {release = resolve;});
      runtime.product.columnsNow = async (...args) => {entered = true; await gate; return read(...args);};
      const promise = runtime.pauseAtDisplayedState();
      for (let i = 0; i < 10 && !entered; i++) await new Promise(resolve => setTimeout(resolve, 0));
      assert.equal(entered, true);
      runtime.control(mode); assert.equal(runtime.running, false); release();
      await assert.rejects(promise, /interrupted by playback/);
      await new Promise(resolve => setTimeout(resolve, 0));
      assert.equal(runtime.handoffPaused, false); assert.equal(runtime.resimming, false);
      assert.equal(runtime.play.mode, mode === "continue" ? "live" : mode);
      assert.equal(runtime.running, mode === "continue");
      assert.equal(runtime.steps, 6);
    }
  });

  await check("cancelled or replaced run cannot publish stale captured state", async () => {
    for (const cancel of [r => r.sync(null), r => {r.token++;}]) {
      const {runtime} = fixture(); runtime.inflight = 1;
      const promise = runtime.pauseAtDisplayedState(); cancel(runtime); runtime.inflight = 0;
      await assert.rejects(promise, /changed during handoff/); assert.equal(runtime.product.readCount, undefined);
    }
  });

  await check("unsupported frame, absent pose and excessive replay refuse without invented state", async () => {
    const {runtime} = fixture();
    await assert.rejects(runtime.pauseAtDisplayedState({maxReplaySteps: 0}), /needs 6 replay steps/);
    assert.equal(runtime.running, true); assert.equal(runtime.handoffPaused, false);
    await assert.rejects(runtime.pauseAtDisplayedState({maxReplaySteps: -1}), /nonnegative/);
    runtime.shown = null; await assert.rejects(runtime.pauseAtDisplayedState(), /No stable/);
    runtime.shown = 99; await assert.rejects(runtime.pauseAtDisplayedState(), /no longer recorded/);
    const missing = fixture().runtime; missing.rotations.delete(3);
    await assert.rejects(missing.pauseAtDisplayedState(), /orientations are unavailable/);
    const divergent = fixture().runtime; divergent.tape.get(3)[0] += 1;
    await assert.rejects(divergent.pauseAtDisplayedState(), /differs from restored native state/);
  });

  await check("new epoch seeds all law columns and recorded orientation before first frame", async () => {
    const {runtime} = fixture();
    runtime.sync({...runtime.spec, epoch: 2, start: encoded}); await runtime.queue;
    assert.equal(runtime.error, null); assert.equal(runtime.epoch, 2);
    for (const name of names) assert.deepEqual(runtime.product.state[name], initial[name], name);
    assert.deepEqual(runtime.rotations.get(0).map(row => row.quaternion), orientations);
    const seeded = runtime.rotations.get(0)[0];
    assert.deepEqual(integrateAngular(seeded.quaternion, [0,0,0], 1), q0);
    assert.equal(runtime.handoffPaused, false);
  });

  await check("actual runtime first-snapshot rotation continues the seeded pose", async () => {
    const tasks = [];
    globalThis.MessageChannel = class {
      constructor() {this.port1 = {}; this.port2 = {postMessage: () => tasks.push(() => this.port1.onmessage())};}
    };
    const {runtime} = fixture();
    const product = runtime.product; product.kind = "wasm"; product.maxFrames = 1; product.canBatch = () => true;
    product.batch = async ({frames, snapEvery}) => {
      const result = {frames: [], spins: [], diags: [], keys: [], emojiSpins: []};
      for (let f = 0; f < frames; f++) {
        product.step(snapEvery); result.frames.push(new Float32Array(10));
        result.spins.push(orientations.map((_, id) => [product.state.wx[id], product.state.wy[id], product.state.wz[id]]));
        result.diags.push(null); result.keys.push(null); result.emojiSpins.push(null);
      }
      return result;
    };
    runtime.sync({...runtime.spec, epoch: 2, start: encoded}); await runtime.queue;
    assert.equal(tasks.length, 1); tasks.shift()();
    await new Promise(resolve => setTimeout(resolve, 0));
    for (let id = 0; id < 2; id++) assert.deepEqual(runtime.rotations.get(1)[id].quaternion,
      integrateAngular(orientations[id], [initial.wx[id], initial.wy[id], initial.wz[id]], snap * dt));
    runtime.sync(null); globalThis.MessageChannel = undefined;
  });

  await check("actual WASM readback includes requested padded state", async () => {
    const product = new WasmProduct(); product.n = n; product.pad = pad; product.inputs = initial; product.nexts = {};
    assert.equal((await product.columnsNow(["age"])).age.length, n);
    assert.deepEqual((await product.columnsNow(["age"], pad)).age, initial.age);
    await assert.rejects(product.columnsNow(["age"], 5), /readback count/);
  });

  await check("actual GPU readback copies padded native column bytes", async () => {
    globalThis.GPUBufferUsage = {MAP_READ: 1, COPY_DST: 2}; globalThis.GPUMapMode = {READ: 1};
    const product = new GpuProduct(); product.n = n; product.pad = pad;
    product.abi = {feed_slots: {age: 0}, output_slots: {}}; product.lanes = [initial.age.buffer];
    product.laneOf = () => 0; product.byteAt = () => 0;
    product.device = {createBuffer({size}) {const bytes = new ArrayBuffer(size); return {bytes, async mapAsync() {}, getMappedRange: () => bytes, unmap() {}, destroy() {}};},
      createCommandEncoder() {return {copyBufferToBuffer(source, offset, target, dest, length) {new Uint8Array(target.bytes, dest, length).set(new Uint8Array(source, offset, length));}, finish() {return {};}};},
      queue: {submit() {}}};
    assert.deepEqual((await product.columnsNow(["age"], pad)).age, initial.age);
    assert.equal((await product.columnsNow(["age"])).age.length, n);
    await assert.rejects(product.columnsNow(["age"], 0), /readback count/);
    delete globalThis.GPUBufferUsage; delete globalThis.GPUMapMode;
  });

  console.log(`${checks} compute state checks passed`);
} finally {
  globalThis.MessageChannel = messageChannel;
  if (path.dirname(temp) !== path.resolve(os.tmpdir()) || !path.basename(temp).startsWith("meowdoku-state-"))
    throw new Error("Unexpected temporary test path");
  await fs.rm(temp, {recursive: true, force: true});
}
