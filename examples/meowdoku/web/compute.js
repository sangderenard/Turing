// The page's generic COMPUTE-PRODUCT runtime.
//
import { caseVibration, caseNudge } from "./case_vibration.js";
// A game declares a compute product in its state:
//
//   game.compute = { manifest, size, epoch, uniforms, speed, named }
//
// `manifest` is a compiler-built products manifest (fountain_build.py writes
// one): per size, the generated WGSL program + its ABI (dispatch entry points
// over ONE arena buffer: a head of uniforms, then lane slots of `count`
// floats; the slot of every formal and output), the wasm product when the
// size has one, the compiler's deployment verdict (`route`, `verdict`), the
// arena's memory and how to present it (`points`, `display_stride`).  This
// runtime creates the device, checks the device's limits against the size's
// memory BEFORE allocating, keeps the arena resident, runs the program's
// dispatches each step, and falls back to the wasm product when the
// classifier said the dispatches do not pay or `navigator.gpu` is missing.
// Nothing here knows what the law is.
//
// Nothing per-id is uploaded: the law derives every id's constants from the
// id, and the generated shader reads the iota from its invocation index.  The
// arena is cleared on the GPU, not filled from the host.
//
// Presentation: the main renderer is WebGL2, which cannot read WebGPU
// buffers.  The product's `points` description (position, velocity for the
// at-rest flag, colour, radius, liveness) is packed on the device by one
// small generic compute pass into 20 bytes per DISPLAYED id -- [x, y, z, rgba8
// (as bits), meta] -- every `display_stride`-th id when the count is large,
// and that packed buffer alone is copied back (triple buffered, one or two
// frames late) for WebGL to draw as GL points in the shared scene.  No WGSL
// is read by a vertex stage, so devices with no vertex-stage storage buffers
// (Apple's) are not a case.  The wasm route packs the same layout on the CPU.
//
// Run modes.  The page's pump steps the law as fast as the device allows, in batches of whole snapshots, NOT from
// requestAnimationFrame: each snapshot (the packed points after `snapEvery` steps) is read back into a bounded RECORDING
// (recording.js: a sliding window under a memory cap, with keyframes of the law's state and a log of the runtime law
// inputs) and the display draws from that buffer at screen rate, at the pace of the simulation's own clock (x1, x2, x4;
// "max" follows the newest).  The GPU queue is bounded (a few batches in flight, the steps per batch adapted to the
// measured step cost) and the pump pauses when the buffer is `LEAD_FRAMES` ahead of the display.  Rebirth restarts from
// the start state, Replay and Bounce play the recorded window forward / forward-and-back (the pump rests), and Continue
// resumes the simulation from the displayed frame -- exactly: the state of that frame is rebuilt from the nearest keyframe
// by running the law again with the logged inputs, which is bit for bit what ran.  The default run is CONTINUOUS (it
// never stops); "settle" is the stop-at-rest rule below, which the image replay needs.
//
// Runtime law inputs (the gravity vector, the lid) are uniforms the page writes between batches, moment by moment, with no
// rebuild; every change is logged with its step so a re-run reproduces it.
//
// Determinism: reset writes the same bytes every time and the law has no
// atomics and no scatter, so a replay from the same initial state on the same
// device repeats exactly.  The runtime records the rest positions when phase
// 1 settles (one readback, an event), compares phase 2's with them, and says
// so on screen; if a device is not bit-identical it records phase 2's own
// positions and replays again, so the image still forms.
import { TuringGraphProgram } from "./turing-wasm-shell.js";
import { Recording, Playback, recordedAngle } from "./recording.js";
import { ParticleStyles, GRAIN_COLOR_GLSL, grainUniforms, decodeMeta } from "./particle_styles.js";
import { NoiseFilter } from "./noise_filter.js";
import { mobroadInputs, mobroadTarget, mobroadPather } from "./mobroad.js";               // [MOBROAD]
import { caseFlipAt, caseFlipSteps, lawAxis, rotateLawVector } from "./case_motion.js";
import { integrateAngular } from "./marble_rotation.js";
import { EmojiTumble, EMOJI_TUMBLE_LIMIT } from "./emoji_tumble.js";
const NO_TINT = new Float32Array(4);     // the colour source "simulation": no override

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

// One displayed id is PT_WORDS 32-bit words: [x, y, z, rgba8 (bits), meta].
const PT_WORDS = 5, PT_BYTES = PT_WORDS * 4;
const mb = (bytes) => (bytes / 1048576).toFixed(0) + " MB";

// The pure sampling step of the image control, exported so a test can hold it against
// fountain.fit_extent + fountain.sample_raster: rest positions -> frame fractions by the
// pile's own extent -> nearest raster pixel -> byte / 255 rounded to f32.
export function sampleRasterAt(raster, X, Y, count, into) {
  const { w, h, data } = raster, n = count ?? X.length;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) { x0 = Math.min(x0, X[i]); x1 = Math.max(x1, X[i]); y0 = Math.min(y0, Y[i]); y1 = Math.max(y1, Y[i]); }
  const bytes = new Uint8Array(3 * n), channels = into.map(() => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    const u = Math.fround((X[i] - x0) / Math.max(x1 - x0, 1e-12)), v = Math.fround(1 - (Y[i] - y0) / Math.max(y1 - y0, 1e-12));
    const px = Math.min(Math.max(Math.floor(u * w), 0), w - 1), py = Math.min(Math.max(Math.floor(v * h), 0), h - 1);
    const at = 4 * (py * w + px);
    for (let c = 0; c < into.length; c++) { bytes[3 * i + c] = data[at + c]; channels[c][i] = Math.fround(data[at + c] / 255); }
  }
  return { bytes, channels };
}

// ---------------------------------------------------------------- GPU route
// The arena is SPLIT across storage bindings (the compiler's `bindings`, `slots_per_binding`): slot s
// lives in lane buffer floor(s / spb) at element (s % spb) * pad, so no single-buffer limit caps the
// size.  The uniforms are their own small `head` buffer.  `ref(slot)` spells "lane<k>[base + i]".
const PACK_WGSL = `
__LANES__
// u32 words, not f32: the colour word is rgba8 bits (alpha 255 puts it in NaN/inf patterns as a float), and a
// store of a NaN pattern through an f32 may be canonicalised on some GPUs (Apple's), losing the colour and alpha
@group(0) @binding(__PTSB__) var<storage, read_write> pts: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
  let j = g.x + g.y * grid.x * 64u;
  if (j >= __NDISP__u) { return; }
  let i = j * __STRIDE__u;
  let live = __LIVE__;
  let vx = __VX__;
  let vy = __VY__;
  let vz = __VZ__;
  let rest = (vx * vx + vy * vy + vz * vz) < __REST2__;
  let c = pack4x8unorm(vec4<f32>(__R__, __G__, __B__, 1.0));
  pts[5u * j] = bitcast<u32>(__X__);
  pts[5u * j + 1u] = bitcast<u32>(__Y__);
  pts[5u * j + 2u] = bitcast<u32>(__Z__);
  pts[5u * j + 3u] = c;
  pts[5u * j + 4u] = bitcast<u32>(select(0.0, select(100.0, 200.0, rest) + __RAD__ * 1000.0, live));
}`;

class GpuProduct {
  kind = "webgpu-compute";

  // Structural limits only (a binding's size, the bindings a stage may use, the workgroup grid).  The
  // total is NOT gated here: it is allocated under an out-of-memory error scope and a real failure is
  // reported.  There is no size gate.
  static async limits(rec) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { ok: false, reason: "no WebGPU adapter" };
    const L = adapter.limits, abi = rec.wgsl_abi;
    const bindingBytes = Math.min(abi.slots_per_binding, abi.slots) * rec.pad * 4;   // the bytes actually bound: the lane holds only the slots it has, not the whole per-binding allowance
    if (bindingBytes > L.maxStorageBufferBindingSize) return { ok: false, reason: `a ${mb(bindingBytes)} binding exceeds this device's maxStorageBufferBindingSize (${mb(L.maxStorageBufferBindingSize)})` };
    if (bindingBytes > L.maxBufferSize) return { ok: false, reason: `a ${mb(bindingBytes)} buffer exceeds this device's maxBufferSize (${mb(L.maxBufferSize)})` };
    if (abi.bindings + 1 > L.maxStorageBuffersPerShaderStage) return { ok: false, reason: `every pass (step, commit, pack) binds ${abi.bindings + 1} storage buffers (${abi.bindings} stacked lanes + the head); the device allows ${L.maxStorageBuffersPerShaderStage} per stage` };
    if (Math.max(abi.groups[0], abi.groups[1]) > L.maxComputeWorkgroupsPerDimension) return { ok: false, reason: `needs ${abi.groups[0]}x${abi.groups[1]} workgroups; the device allows ${L.maxComputeWorkgroupsPerDimension} per dimension` };
    return { ok: true, adapter, limits: L, bindingBytes };
  }

  async init(rec, manifest, angularCount = 1) {
    const check = await GpuProduct.limits(rec);
    if (!check.ok) throw new Error("refused: " + check.reason);
    const adapter = check.adapter, A = rec.wgsl_abi, B = A.bindings;
    // request no more than the adapter reports (an over-ask makes requestDevice itself fail, e.g. on a phone)
    const AL = adapter.limits, ask = (name, want) => Math.min(AL[name], want);
    this.device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: ask("maxStorageBufferBindingSize", Math.max(check.bindingBytes, 134217728)),
      maxBufferSize: ask("maxBufferSize", Math.max(check.bindingBytes, 268435456)),
      maxStorageBuffersPerShaderStage: ask("maxStorageBuffersPerShaderStage", Math.max(B + 1, 8)) } });
    this.device.addEventListener?.("uncapturederror", (e) => { this.failure = String((e.error && e.error.message) || e.error); });
    this.device.lost?.then((i) => { this.failure = `device lost (${i.reason}): ${i.message}`; });
    this.scopeFaults = 0;
    this.adapterInfo = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.device].filter(Boolean).join("/") : "";
    this.rec = rec; this.n = rec.n; this.pad = rec.pad; this.abi = A; this.manifest = manifest;
    this.state = manifest.law.state;
    const pad = this.pad, spb = A.slots_per_binding;
    this.laneOf = (slot) => Math.floor(slot / spb);
    this.byteAt = (slot) => (slot % spb) * pad * 4;                     // byte offset of a lane slot in its buffer
    this.ref = (slot) => `lane${Math.floor(slot / spb)}[${(slot % spb) * pad}u + i]`;
    const lanes = (access) => Array.from({ length: B }, (_, b) => `@group(0) @binding(${b}) var<storage, ${access}> lane${b}: array<f32>;`).join("\n");
    const code = await fetch(rec.wgsl).then((r) => r.text());
    const module = this.device.createShaderModule({ code });
    this.#check(await module.getCompilationInfo(), "generated WGSL");
    const entries = Array.from({ length: B + 1 }, (_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }));
    const layout = this.device.createBindGroupLayout({ entries });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipelines = [];
    for (let k = 0; k < A.entries.length; k += 64) {                 // in batches: hundreds of entry points at the largest sizes
      this.pipelines.push(...await Promise.all(A.entries.slice(k, k + 64).map((entryPoint) =>
        this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } }))));
    }
    const slot = (name) => A.feed_slots[name], oslot = (name) => A.output_slots[name], p = manifest.points;
    const stride = rec.display_stride, ndisp = rec.memory.display_ids;
    // the state commit: each state output over its feed slot, one more dispatch (WebGPU forbids a copy within one buffer)
    const commit = this.state.map((name) => `  ${this.ref(slot(name))} = ${this.ref(oslot(name + "_next"))};`).join("\n");
    const commitModule = this.device.createShaderModule({ code: `${lanes("read_write")}
@group(0) @binding(${B}) var<storage, read_write> head: array<f32>;
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
  let i = g.x + g.y * grid.x * 64u;
  if (i >= ${pad}u) { return; }
${commit}
}` });
    this.#check(await commitModule.getCompilationInfo(), "commit WGSL");
    this.pipelines.push(await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: commitModule, entryPoint: "commit" } }));
    // the pack pass reads element j * stride of a slot, not i
    const at = (s) => `lane${Math.floor(s / spb)}[${(s % spb) * pad}u + i]`;
    const packCode = PACK_WGSL.replace("__LANES__", lanes("read_write")).replace("__PTSB__", String(B))
      .replaceAll("__NDISP__", String(ndisp)).replaceAll("__STRIDE__", String(stride))
      .replaceAll("__LIVE__", p.placed ? `(abs(${at(slot(p.placed[0]))}) + abs(${at(slot(p.placed[1]))})) >= 1.0e-30`   // 2-D product: placed (not the parked zero state), whatever the nozzle rate
        : Array.isArray(p.live) ? `${at(slot(p.live[0]))} >= f32(i) * ${rec.emit_rate.toPrecision(9)}f` : `${at(oslot(p.live))} > 0.5`)   // else emitted by age; 3-D: its `shown` output
      .replaceAll("__X__", at(slot(p.x))).replaceAll("__Y__", at(slot(p.y))).replaceAll("__Z__", p.z === undefined ? "0.0" : at(slot(p.z)))       // a 2-D product has no z: 0
      .replaceAll("__VX__", at(slot(p.vx))).replaceAll("__VY__", at(slot(p.vy))).replaceAll("__VZ__", p.vz === undefined ? "0.0" : at(slot(p.vz)))
      .replaceAll("__R__", at(oslot(p.rgb[0]))).replaceAll("__G__", at(oslot(p.rgb[1]))).replaceAll("__RAD__", at(oslot(p.radius)))
      .replaceAll("__B__", at(oslot(p.rgb[2]))).replaceAll("__REST2__", (p.rest_speed * p.rest_speed).toFixed(8));
    const pack = this.device.createShaderModule({ code: packCode });
    this.#check(await pack.getCompilationInfo(), "pack WGSL");
    const packEntries = Array.from({ length: B + 1 }, (_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }));
    const packLayout = this.device.createBindGroupLayout({ entries: packEntries });
    this.pack = await this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [packLayout] }), compute: { module: pack, entryPoint: "main" } });
    this.ndisp = ndisp;
    // allocation, with out-of-memory caught: a real failure reported, never a size gate
    this.device.pushErrorScope("out-of-memory");
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const lastSlots = A.slots - spb * (B - 1);
    this.lanes = Array.from({ length: B }, (_, b) => this.device.createBuffer({ size: (b === B - 1 ? lastSlots : spb) * pad * 4, usage: S }));
    this.head = this.device.createBuffer({ size: Math.max(16, Math.ceil(Object.keys(A.uniform_index).length / 4) * 16), usage: S });
    this.pts = this.device.createBuffer({ size: ndisp * PT_BYTES, usage: S });
    // Snapshot batches are read back through a few staging buffers: each holds up to `maxFrames` packed snapshots and, ahead of
    // them, the law's diagnostics after the snapshots that asked for them.  (32 MB at most per staging buffer.)
    this.frameBytes = ndisp * PT_BYTES; this.diagNames = manifest.law.diagnostics; this.pool = [];
    this.maxFrames = Math.max(1, Math.min(16, Math.floor(33554432 / this.frameBytes)));
    this.diagRegion = Math.ceil(this.maxFrames * this.diagNames.length * 4 / 256) * 256;
    this.angularSlots = ["wx", "wy", "wz"].every((name) => A.feed_slots[name] !== undefined)
      ? ["wx", "wy", "wz"].map((name) => A.feed_slots[name]) : null;
    this.angularCount = this.angularSlots ? Math.min(rec.n, angularCount) : 0;
    this.angularRegion = this.angularSlots ? Math.ceil(this.maxFrames * this.angularCount * 12 / 256) * 256 : 0;
    this.emojiAngularCount = this.angularSlots ? Math.min(ndisp, EMOJI_TUMBLE_LIMIT) : 0;
    this.emojiRegion = Math.ceil(this.maxFrames * this.emojiAngularCount * 12 / 256) * 256;
    this.emojiOffset = this.diagRegion + this.angularRegion;
    this.frameOffset = this.emojiOffset + this.emojiRegion;
    if (this.emojiAngularCount) {
      const spinCode = `${lanes("read_write")}
@group(0) @binding(${B}) var<storage, read_write> spin: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= ${this.emojiAngularCount}u) { return; }
  let i = g.x * ${stride}u;
  spin[g.x * 3u] = ${this.ref(this.angularSlots[0])};
  spin[g.x * 3u + 1u] = ${this.ref(this.angularSlots[1])};
  spin[g.x * 3u + 2u] = ${this.ref(this.angularSlots[2])};
}`;
      const spinModule = this.device.createShaderModule({ code: spinCode });
      this.#check(await spinModule.getCompilationInfo(), "emoji angular gather WGSL");
      this.emojiPack = await this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [packLayout] }), compute: { module: spinModule, entryPoint: "main" } });
      this.emojiBuffer = this.device.createBuffer({ size: this.emojiAngularCount * 12, usage: S });
    }
    this.stages = [0, 1, 2].map(() => ({ buffer: this.device.createBuffer({ size: this.frameOffset + this.maxFrames * this.frameBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
    this.stateSlots = this.state.map((name) => A.feed_slots[name]);
    const oom = await this.device.popErrorScope();
    if (oom) throw new Error(`out of memory: the device could not allocate ${mb(rec.memory.arena_bytes + ndisp * PT_BYTES)} for ${rec.n.toLocaleString()} ids (${oom.message})`);
    this.group = this.device.createBindGroup({ layout, entries: [...this.lanes.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: B, resource: { buffer: this.head } }] });
    this.packGroup = this.device.createBindGroup({ layout: packLayout, entries: [...this.lanes.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: B, resource: { buffer: this.pts } }] });
    if (this.emojiBuffer) this.emojiGroup = this.device.createBindGroup({ layout: packLayout, entries: [...this.lanes.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: B, resource: { buffer: this.emojiBuffer } }] });
    this.stepMs = 0;
  }

  #check(info, what) {
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(`${what} rejected: ` + errors.map((m) => `${m.lineNum}: ${m.message}`).join(" | "));
  }

  // zero on the GPU (no host array), then the uniforms and the per-id image colours
  reset({ named, uniforms }) {
    const encoder = this.device.createCommandEncoder();
    for (const buffer of [...this.lanes, this.head]) encoder.clearBuffer(buffer);
    this.device.queue.submit([encoder.finish()]);
    for (const [name, value] of Object.entries(uniforms)) this.setUniform(name, value);
    this.setColours(named);
  }

  setUniform(name, value) { const k = this.abi.uniform_index[name]; if (k !== undefined) this.device.queue.writeBuffer(this.head, k * 4, new Float32Array([value])); }   // a name this product has no uniform for (grav_z on the 2-D one) is ignored

  setColours(named) {
    for (const [name, values] of Object.entries(named)) {
      const slot = this.abi.feed_slots[name];
      if (slot !== undefined) this.device.queue.writeBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), values);
    }
  }

  #encodeSteps(encoder, count) {
    const g = this.abi.groups;
    for (let k = 0; k < count; k++) {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, this.group);
      for (const pipeline of this.pipelines) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(g[0], g[1], 1); }
      pass.end();
    }
  }

  #encodePack(encoder) {
    const pack = encoder.beginComputePass();
    pack.setPipeline(this.pack); pack.setBindGroup(0, this.packGroup);
    const tg = Math.ceil(this.ndisp / 64), gx = Math.min(tg, 16384);
    pack.dispatchWorkgroups(gx, Math.ceil(tg / gx), 1);
    pack.end();
  }

  #scoped(submit) {
    // validation / internal errors of every submission surface in the status line (failure), not only uncaptured ones
    for (const kind of ["validation", "internal"]) this.device.pushErrorScope(kind);
    submit();
    for (const kind of ["internal", "validation"]) this.device.popErrorScope().then((e) => { if (e) { this.scopeFaults++; this.failure = `${kind}: ${e.message}`; } });
  }

  canBatch() { return this.stages.some((s) => !s.busy); }

  // Run `frames` snapshots: `snapEvery` steps, then pack and keep the points, each; plus the law's diagnostics after the
  // snapshots in `diagAt` and a copy of the law's state after those in `keyAt` (indices into the batch).  One submit.  Returns
  // a promise of { frames: Float32Array[], diags: ({name: value} | null)[], keys: ({ payload, bytes } | null)[] }.
  batch({ frames, snapEvery, diagAt = [], keyAt = [], emojiSpin = false }) {
    const stage = this.stages.find((s) => !s.busy);
    if (!stage) throw new Error("no staging buffer free");
    stage.busy = true;
    const nd = this.diagNames.length, keys = new Array(frames).fill(null);
    const encoder = this.device.createCommandEncoder();
    for (let j = 0; j < frames; j++) {
      this.#encodeSteps(encoder, snapEvery);
      this.#encodePack(encoder);
      encoder.copyBufferToBuffer(this.pts, 0, stage.buffer, this.frameOffset + j * this.frameBytes, this.frameBytes);
      if (this.angularSlots) for (let id = 0; id < this.angularCount; id++) this.angularSlots.forEach((slot, k) =>
        encoder.copyBufferToBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot) + id * 4, stage.buffer,
          this.diagRegion + (j * this.angularCount + id) * 12 + k * 4, 4));
      if (emojiSpin && this.emojiPack) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.emojiPack); pass.setBindGroup(0, this.emojiGroup);
        pass.dispatchWorkgroups(Math.ceil(this.emojiAngularCount / 64)); pass.end();
        encoder.copyBufferToBuffer(this.emojiBuffer, 0, stage.buffer, this.emojiOffset + j * this.emojiAngularCount * 12, this.emojiAngularCount * 12);
      }
      if (diagAt.includes(j)) this.diagNames.forEach((name, k) => {
        const slot = this.abi.output_slots[name];
        encoder.copyBufferToBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), stage.buffer, (j * nd + k) * 4, 4);
      });
      if (keyAt.includes(j)) {
        const buffer = this.device.createBuffer({ size: this.stateSlots.length * this.pad * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.stateSlots.forEach((slot, k) => encoder.copyBufferToBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), buffer, k * this.pad * 4, this.pad * 4));
        keys[j] = { payload: buffer, bytes: buffer.size };
      }
    }
    this.#scoped(() => this.device.queue.submit([encoder.finish()]));
    const used = this.frameOffset + frames * this.frameBytes;
    return stage.buffer.mapAsync(GPUMapMode.READ, 0, used).then(() => {
      const range = stage.buffer.getMappedRange(0, used), words = this.frameBytes / 4;
      const out = { frames: [], diags: [], keys, spins: [], emojiSpins: [] };
      for (let j = 0; j < frames; j++) {
        const dst = this.pool.pop() || new Float32Array(words);
        dst.set(new Float32Array(range, this.frameOffset + j * this.frameBytes, words));
        out.frames.push(dst);
        out.spins.push(this.angularSlots ? Array.from({ length: this.angularCount }, (_, id) =>
          Array.from(new Float32Array(range, this.diagRegion + (j * this.angularCount + id) * 12, 3))) : null);
        out.emojiSpins.push(emojiSpin && this.emojiPack
          ? new Float32Array(new Float32Array(range, this.emojiOffset + j * this.emojiAngularCount * 12, this.emojiAngularCount * 3)) : null);
        out.diags.push(diagAt.includes(j) ? Object.fromEntries(this.diagNames.map((name, k) => [name, new Float32Array(range, (j * nd + k) * 4, 1)[0]])) : null);
      }
      stage.buffer.unmap(); stage.busy = false;
      return out;
    }).catch((e) => { stage.busy = false; this.failure = "readback: " + String((e && e.message) || e); throw e; });
  }

  // advance `count` steps with no snapshot (test hook, and re-running the law to rebuild a frame)
  step(count) {
    for (let done = 0; done < count; done += 32) {
      const encoder = this.device.createCommandEncoder();
      this.#encodeSteps(encoder, Math.min(32, count - done));
      this.#scoped(() => this.device.queue.submit([encoder.finish()]));
    }
  }

  // the law's state (the keyframe payload) back into the lanes
  restore(buffer) {
    const encoder = this.device.createCommandEncoder();
    this.stateSlots.forEach((slot, k) => encoder.copyBufferToBuffer(buffer, k * this.pad * 4, this.lanes[this.laneOf(slot)], this.byteAt(slot), this.pad * 4));
    this.device.queue.submit([encoder.finish()]);
  }
  freeKey(payload) { payload.destroy(); }

  // an event, never per frame: named columns, the first `n` ids of each
  async columnsNow(names) {
    const out = {}, n = this.n;
    for (const name of names) {
      const slot = this.abi.output_slots[name] !== undefined && this.abi.feed_slots[name] === undefined ? this.abi.output_slots[name] : this.abi.feed_slots[name];
      const read = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), read, 0, n * 4);
      this.device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      out[name] = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
    }
    return out;
  }

  // The law's DIAGNOSTICS (energy, largest speed, the id counts, dropped contacts): element 0 of each reduced output,
  // read in one small copy.  Call when the queue is idle -- it reads the last completed step.
  async diagNow() {
    const names = this.manifest.law.diagnostics, read = this.device.createBuffer({ size: names.length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    names.forEach((name, k) => { const slot = this.abi.output_slots[name]; encoder.copyBufferToBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), read, k * 4, 4); });
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
    return Object.fromEntries(names.map((name, k) => [name, v[k]]));
  }

  // [LIVEVIEW] write a whole named state column (the first `n` ids): a viewer taking the owner's position keyframe.  Same queue
  // write as `nudge` below, in order with the steps already submitted.
  writeColumn(name, values) {
    const slot = this.abi.feed_slots[name];
    if (slot === undefined || values.length !== this.n) return false;
    this.device.queue.writeBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), values);
    return true;
  }

  nudge(ids, dx) {              // test hook: perturb a few ids' x, to exercise the replay-not-identical path
    const slot = this.abi.feed_slots.x;
    for (const id of ids) this.device.queue.writeBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot) + id * 4, new Float32Array([dx]));
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); }
  destroy() { for (const b of [...(this.lanes || []), this.head, this.pts, this.emojiBuffer]) b?.destroy(); for (const s of this.stages || []) s.buffer.destroy(); this.device.destroy(); }
}

// ---------------------------------------------------------------- wasm route
class WasmProduct {
  kind = "wasm";

  async init(rec, manifest, angularCount = 1) {
    this.rec = rec; this.n = rec.n; this.pad = rec.pad; this.manifest = manifest;
    this.program = await TuringGraphProgram.load(rec.wasm, rec.wasm_abi);
    const abi = this.program.abi;
    this.inputs = {}; this.nexts = {};
    for (const name of abi.argument_names) this.inputs[name] = this.program.view(abi.inputs[name]);
    for (const out of abi.outputs) this.nexts[out.name] = this.program.view(out.id);
    this.state = manifest.law.state;
    this.angularCount = Math.min(rec.n, angularCount);
    this.ndisp = this.pad; this.frameBytes = this.pad * PT_BYTES; this.pool = []; this.maxFrames = 16;
  }

  reset({ named, uniforms }) {
    for (const view of Object.values(this.inputs)) if (view.length > 1) view.fill(0);
    for (let i = 0; i < this.pad; i++) this.inputs.pid[i] = i;
    for (const [name, values] of Object.entries(named)) if (this.inputs[name] && this.inputs[name].length >= values.length) this.inputs[name].set(values);
    for (const [name, value] of Object.entries(uniforms)) if (this.inputs[name]) this.inputs[name][0] = value;
  }

  setUniform(name, value) { if (this.inputs[name]) this.inputs[name][0] = value; }
  setColours(named) { for (const [name, values] of Object.entries(named)) this.inputs[name].set(values); }

  step(count) {
    for (let k = 0; k < count; k++) {
      this.program.run();
      for (const name of this.state) this.inputs[name].set(this.nexts[name + "_next"]);
    }
  }

  canBatch() { return true; }

  // the same batch as the GPU's, run now on this thread (see GpuProduct.batch)
  batch({ frames, snapEvery, diagAt = [], keyAt = [], emojiSpin = false }) {
    const out = { frames: [], diags: [], keys: [], spins: [], emojiSpins: [] };
    for (let j = 0; j < frames; j++) {
      this.step(snapEvery);
      out.frames.push(this.#pack(this.pool.pop() || new Float32Array(this.pad * PT_WORDS)));
      out.spins.push(["wx", "wy", "wz"].every((name) => this.inputs[name])
        ? Array.from({ length: this.angularCount }, (_, id) => [this.inputs.wx[id], this.inputs.wy[id], this.inputs.wz[id]]) : null);
      if (emojiSpin && ["wx", "wy", "wz"].every((name) => this.inputs[name])) {
        const count = Math.min(this.ndisp, EMOJI_TUMBLE_LIMIT), spin = new Float32Array(count * 3), stride = this.rec.display_stride;
        for (let id = 0; id < count; id++) {
          spin[id * 3] = this.inputs.wx[id * stride]; spin[id * 3 + 1] = this.inputs.wy[id * stride]; spin[id * 3 + 2] = this.inputs.wz[id * stride];
        }
        out.emojiSpins.push(spin);
      } else out.emojiSpins.push(null);
      out.diags.push(diagAt.includes(j) ? Object.fromEntries(this.manifest.law.diagnostics.map((name) => [name, this.nexts[name][0]])) : null);
      out.keys.push(keyAt.includes(j) ? { payload: Object.fromEntries(this.state.map((name) => [name, this.inputs[name].slice()])), bytes: this.state.length * this.pad * 4 } : null);
    }
    return Promise.resolve(out);
  }

  restore(payload) { for (const name of this.state) this.inputs[name].set(payload[name]); }
  freeKey() {}

  #pack(f) {
    const p = this.manifest.points, I = this.inputs, O = this.nexts, u = new Uint32Array(f.buffer, f.byteOffset, f.length);
    const x = I[p.x], y = I[p.y], vx = I[p.vx], vy = I[p.vy], rate = this.rec.emit_rate;
    const z = p.z === undefined ? null : I[p.z], vz = p.vz === undefined ? null : I[p.vz];
    const age = Array.isArray(p.live) && !p.placed ? I[p.live[0]] : null, shown = Array.isArray(p.live) ? null : O[p.live];
    const px = p.placed ? I[p.placed[0]] : null, py = p.placed ? I[p.placed[1]] : null;
    const r = O[p.rgb[0]], g = O[p.rgb[1]], b = O[p.rgb[2]], rad = O[p.radius], rest2 = p.rest_speed * p.rest_speed;
    const byte = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
    for (let i = 0; i < this.pad; i++) {
      f[5 * i] = x[i]; f[5 * i + 1] = y[i]; f[5 * i + 2] = z ? z[i] : 0;
      u[5 * i + 3] = (255 << 24) | (byte(b[i]) << 16) | (byte(g[i]) << 8) | byte(r[i]);
      const on = px ? Math.abs(px[i]) + Math.abs(py[i]) >= 1e-30 : age ? age[i] >= i * rate : shown[i] > 0.5, v2 = vx[i] * vx[i] + vy[i] * vy[i] + (vz ? vz[i] * vz[i] : 0);
      f[5 * i + 4] = on ? (v2 < rest2 ? 200 : 100) + rad[i] * 1000 : 0;
    }
    return f;
  }

  async columnsNow(names) {
    const n = this.n;
    return Object.fromEntries(names.map((name) => [name, Float32Array.from((this.inputs[name] || this.nexts[name]).subarray(0, n))]));
  }
  async diagNow() { return Object.fromEntries(this.manifest.law.diagnostics.map((name) => [name, this.nexts[name][0]])); }
  nudge(ids, dx) { for (const id of ids) this.inputs.x[id] = dx; }
  async idle() {}
  destroy() {}
}

// ------------------------------------------------------------------ settling
// Phase 1 ends when the pile is AT REST, decided by the law's own energy (a fixed-order reduction on the device), read
// at fixed step multiples (`sample`): everything is emitted, and the kinetic + rotational energy stays under both its
// absolute floor (per id) and a fraction of its peak, with the largest speed under a limit, for `hold_samples` samples in
// a row.  The step of the sample that completes the hold is S; phase 2 stops at exactly S.  `fountain.SettleDetector`
// is the same rule in Python (held against this one by tests/test_meowdoku_fountain.py).
// Initial state columns: Studio can supply a packed float32 age column to
// bound the active ids of an existing 4k product without changing its law.
// The glyph start's placed grains (game.compute.start, glyph_start.GlyphStart.payload): x y z (and rs for a swelling species) as 16-bit fixed
// point over `bounds`, decoded to float32 columns of the product's length.  They are fed beside the image colours (reset({named})): the
// run starts from these positions at rest.  Python's glyph_start.decode_payload is this formula.
export function decodeStart(start, pad) {
  const bytes = (text) => { const bin = atob(text), u8 = new Uint8Array(bin.length); for (let i = 0; i < u8.length; i++) u8[i] = bin.charCodeAt(i); return new Uint16Array(u8.buffer); };
  if (start.age_f32) {
    const raw = atob(start.age_f32);
    if (raw.length !== pad * 4) throw new Error("initial age column length differs from compiled particle count");
    const data = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) data[i] = raw.charCodeAt(i);
    return { age: new Float32Array(data.buffer) };
  }
  if (start.ids) return decodeIdStart(start, pad, bytes);        // TIP JAR: a start that names the ids its rows belong to
  const q = bytes(start.xyz), b = start.bounds, out = {};
  ["x", "y", "z"].forEach((name, c) => {
    const column = new Float32Array(pad), lo = b[2 * c], span = b[2 * c + 1] - lo;
    for (let i = 0; i < start.n; i++) column[i] = lo + q[3 * i + c] / 65535 * span;
    out[name] = column;
  });
  if (start.rs) {
    const r = bytes(start.rs), column = new Float32Array(pad);
    for (let i = 0; i < start.n; i++) column[i] = r[i] / 65535 * start.rmax;
    out.rs = column;
  }
  return out;
}

// TIP JAR (tipjar.py `decode_start` is this formula): the start's row k belongs to id ids[k]; x y z rs, a live `age` and the per-id colours
// ir ig ib (bytes, rgb) are scattered to their ids.  Every other id keeps the exact zero state (parked).
function decodeIdStart(start, pad, bytes) {
  const n = start.n, ids = bytes(start.ids), q = bytes(start.xyz), r = bytes(start.rs), b = start.bounds, out = {};
  const bin = atob(start.rgb), rgb = new Uint8Array(bin.length); for (let i = 0; i < rgb.length; i++) rgb[i] = bin.charCodeAt(i);
  for (const name of ["x", "y", "z", "rs", "age", "ir", "ig", "ib"]) out[name] = new Float32Array(pad);
  for (let k = 0; k < n; k++) {
    const id = ids[k];
    ["x", "y", "z"].forEach((name, c) => { out[name][id] = b[2 * c] + q[3 * k + c] / 65535 * (b[2 * c + 1] - b[2 * c]); });
    out.rs[id] = r[k] / 65535 * start.rmax; out.age[id] = start.age;
    out.ir[id] = rgb[3 * k] / 255; out.ig[id] = rgb[3 * k + 1] / 255; out.ib[id] = rgb[3 * k + 2] / 255;
  }
  return out;
}

// PURRLINE SAND (WORKLIST.md, "Purrline sand mode", C): the game's state as the purrline variant's runtime inputs (fountain_purrline.manifest.json).
// This is fountain.py's ``purrline_inputs`` (a test holds the two equal).  ``game`` is the purrline state (occupancy, rows, cols, cell, origin, falling:
// board units, x right, y DOWN); ``map`` the chamber's board map (the board's centre cx, cy in fountain units and one board cell's width); ``boxes`` the
// law's moving-box count.  The occupied cells are the grid's mask (three words: row r is byte r % 3 of word floor(r / 3)); the lowest falling items are the
// moving boxes (the size of the cell they will fill, centred on the item, carrying its velocity, y flipped); unused boxes are written off.
export const PURRLINE_ROUND = 0.12;
export function purrlineInputs(game, map, boxes) {
  const cb = game.cell ?? 1, [ox, oy] = game.origin || [0, 0], rows = game.rows, cols = game.cols;
  const k = map.cell / cb, bcx = ox + 0.5 * cols * cb, bcy = oy + 0.5 * rows * cb, words = [0, 0, 0], out = {};
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (game.occupancy[r * cols + c]) words[Math.floor(r / 3)] += 2 ** (8 * (r % 3) + c);
  Object.assign(out, { og_on: 1, og_cx: map.cx, og_cy: map.cy, og_cell: map.cell, og_cols: cols, og_rows: rows, og_w0: words[0], og_w1: words[1], og_w2: words[2] });
  const items = [...(game.falling || [])].sort((a, b) => b.y - a.y).slice(0, boxes), half = 0.5 * map.cell;
  for (let i = 0; i < boxes; i++) {
    const it = items[i];
    const v = it
      ? { on: 1, cx: map.cx + (it.x - bcx) * k, cy: map.cy - (it.y - bcy) * k, hw: half, hh: half, rc: PURRLINE_ROUND * map.cell, vx: (it.vx || 0) * k, vy: -(it.vy || 0) * k }
      : { on: 0, cx: 0, cy: 0, hw: 0, hh: 0, rc: 0, vx: 0, vy: 0 };
    for (const [name, value] of Object.entries(v)) out[`mb${i}_${name}`] = value;
  }
  for (const name of Object.keys(out)) out[name] = Math.fround(out[name]);
  return out;
}

export class SettleDetector {
  constructor(p, active, pad) { this.p = p; this.active = active; this.pad = pad; this.peak = 0; this.quiet = 0; }
  push(step, d) {
    this.peak = Math.max(this.peak, d.ke);
    const emitted = d.n_parked <= this.pad - this.active;
    const quiet = emitted && d.ke <= this.p.ke_per_id * this.active && d.ke <= this.p.relative * this.peak && Math.sqrt(d.vmax2) <= this.p.speed;
    this.quiet = quiet ? this.quiet + 1 : 0;
    return this.quiet >= this.p.hold_samples ? step : null;
  }
}

// ------------------------------------------------------------------ runtime
const MAX_IN_FLIGHT = 2;               // batches submitted to the device whose snapshots have not come back (the queue never runs away)
// the gravity of a case spinning at ``hz`` revolutions per second, ``t`` simulated seconds in: straight down at t = 0,
// turning counterclockwise for hz > 0 (fractions of g in the screen's plane)
// ``motion.phase`` (degrees) offsets the cycle: a spin starts turned by it, a rock starts that far into its swing.
export function gravityAngle(hz, t, motion = { kind: "spin" }) {
  const phase = 2 * Math.PI * hz * t + (Math.PI / 180) * (Number(motion.phase) || 0);
  return motion.kind === "rock" ? (Math.PI / 180) * (motion.arc / 2) * Math.sin(phase) : phase;     // theta(t) of the case: spin = phase, rock = (arc/2) sin(phase)
}
export function spinGravity(hz, t, motion = { kind: "spin" }) {
  const a = gravityAngle(hz, t, motion);
  return [Math.sin(a), -Math.cos(a)];
}
// The case's angular velocity and acceleration (rad/s, rad/s^2, counterclockwise positive) behind that gravity: gravity seen in
// the case frame turns by the OPPOSITE of the case's turn, so a spin (gravity angle a = 2 pi hz t) is a case at omega = -da/dt
// and a rock (a = (arc/2) sin(phase)) has omega = -da/dt, domega = -d2a/dt2.  The law adds the fictitious forces for these (Coriolis,
// centrifugal, Euler), so a turning case drags its contents even with gravity at 0.
export function spinOmega(hz, t, motion = { kind: "spin" }) {
  const w = 2 * Math.PI * hz, phase = w * t + (Math.PI / 180) * (Number(motion.phase) || 0);
  if (motion.kind === "rock") {
    const A = (Math.PI / 180) * (motion.arc / 2);
    return [-A * w * Math.cos(phase), A * w * w * Math.sin(phase)];
  }
  return [-w, 0];
}
const SPIN_TURN = 1 / 120;              // the most a spinning case turns between input updates (3 degrees)
const TARGET_MS = { gpu: 8, wasm: 6 };  // what one batch should take: short enough to keep the display and the page responsive
const LEAD_FRAMES = 8;                  // the pump rests when the buffer holds this many snapshots the display has not reached
const DIAG_MS = 400;                    // continuous runs read the law's diagnostics this often (settle reads every sample)

export class ComputeRuntime {
  constructor({ onStatus, onSamples, onSupport }) {
    this.onStatus = onStatus; this.onSamples = onSamples; this.onSupport = onSupport || (() => {});
    this.spec = null; this.manifest = null; this.product = null; this.productKey = null;
    this.epoch = null; this.uniforms = {}; this.support = null;
    this.named = {}; this.start = null; this.raster = null; this.sampleSpec = null; this.rest = null; this.verdict = null;
    this.restPending = false; this.stepMs = 0; this.recovering = false;
    this.steps = 0; this.running = false; this.last = 0; this.fps = 0; this.frames = 0; this.fpsAt = 0;
    this.route = ""; this.why = ""; this.warn = false; this.queue = Promise.resolve(); this.error = null;
    this.ready = false; this.note = "";
    this.settleStep = null; this.detector = null; this.diag = null;
    // the pump, the recording and the display cursor
    this.tape = null; this.play = new Playback(); this.shown = null; this.token = 0; this.inflight = 0; this.arrived = new Map();
    this.rotations = new Map(); // per-id physical omega and integrated pose, keyed with the recording
    this.emojiTumble = new EmojiTumble(); this.emojiTumbleOn = false;
    this.kicked = false; this.channel = null; this.lastDone = 0; this.diagAt = 0; this.resimming = false;
    this.stepsDone = 0; this.stepsPerSec = 0;
    this.lawInputs = {}; this.gravity = null; this.vibration = null; this.nudge = null; this.capBytes = (typeof navigator !== "undefined" && navigator.maxTouchPoints > 0 ? 64 : 192) * 1048576;
  }

  get active() { return !!this.spec; }
  get rec() { return this.manifest && this.spec ? this.manifest.sizes[this.spec.size] : null; }
  get snap() { return this.tape.snapEvery; }
  get settleMode() { return !!(this.uniforms.use_image || (this.spec && this.spec.run === "settle")); }
  get runtimeInputs() { return new Set((this.manifest && this.manifest.law.runtime_inputs) || []); }

  // Called with each game state: the declared product (or null).
  sync(spec) {
    if (!spec) { this.spec = null; this.running = false; return; }
    this.spec = spec;
    this.queue = this.queue.then(() => this.#apply(spec)).catch((error) => {
      this.error = String(error.message || error); this.ready = false; this.running = false; this.onStatus();
      console.error("[compute]", error);
    });
  }

  // What each size needs and whether this device (or the wasm route) can hold it.
  async #capabilities() {
    const support = {};
    for (const [tag, rec] of Object.entries(this.manifest.sizes)) {
      const wasmOk = !!rec.wasm;
      if (rec.route === "webgpu-compute" || !wasmOk) {
        if (!navigator.gpu) { support[tag] = wasmOk ? { ok: true } : { ok: false, reason: "needs WebGPU (this browser has none)" }; continue; }
        const check = await GpuProduct.limits(rec);
        support[tag] = check.ok ? { ok: true } : (wasmOk ? { ok: true, note: "wasm: " + check.reason } : { ok: false, reason: check.reason });
      } else support[tag] = { ok: true };
    }
    this.support = support;
    this.onSupport();
  }

  async #apply(spec) {
    // a game may switch products (fountain3d: the flat product for the box and polygons, the sdf one for SDF chambers): the manifest
    // follows the spec's name
    if (!this.manifest || this.manifestName !== spec.manifest) {
      this.manifest = await json(spec.manifest); this.manifestName = spec.manifest; await this.#capabilities();
    }
    const angularCount = Math.max(1, Math.min(16, Math.floor(spec.studio_count || 1)));
    const key = `${spec.manifest}:${spec.size}:angular${angularCount}`;
    const verdict = this.support[spec.size];
    if (key !== this.productKey) {
      if (verdict && !verdict.ok) throw new Error(`${spec.size} refused: ${verdict.reason}`);
      await this.#load(spec, key);
      this.epoch = null;
    }
    if (spec.epoch !== this.epoch) {
      this.epoch = spec.epoch; this.uniforms = { ...spec.uniforms };
      this.start = spec.start ? decodeStart(spec.start, this.rec.pad) : null;       // a glyph start: the placed grains
      this.#reset();
    } else {
      // the runtime inputs belong to the page from the first reset on: the server's numbers for them are only defaults
      const mine = this.runtimeInputs;
      for (const [name, value] of Object.entries(spec.uniforms)) {
        if (mine.has(name)) continue;
        if (this.uniforms[name] !== value) { this.uniforms[name] = value; this.product.setUniform(name, value); }
      }
    }
  }

  async #load(spec, key) {
    this.ready = false; this.running = false; this.error = null; this.token++;
    // A prior batch may still be mapping a staging buffer. The invalidated
    // batch releases its frames in #land; only then may we destroy its device.
    while (this.inflight > 0) await new Promise((resolve) => setTimeout(resolve, 2));
    if (this.product) { if (this.tape) this.#release(this.tape.clear()); this.product.destroy(); this.product = null; }
    const rec = this.manifest.sizes[spec.size];
    // The compiler decides (rec.route, from its verdict); the browser can only veto.
    const v = rec.verdict, model = this.manifest.cost_model;
    const detail = `n=${rec.n.toLocaleString()}, ${rec.wgsl_abi.dispatches} dispatches, ${mb(rec.memory.arena_bytes)}: predicted ${v.speedup.toFixed(2)}x, needs ${model.min_speedup.toFixed(1)}x`;
    let product = null; this.warn = false;
    if ((rec.route === "webgpu-compute" || !rec.wasm) && navigator.gpu) {
      try { product = new GpuProduct(); await product.init(rec, this.manifest, Math.max(1, Math.min(16, Math.floor(spec.studio_count || 1))));
        this.why = rec.route === "webgpu-compute" ? `compiler verdict: the dispatches pay (${detail})` : `WebGPU product used while the CPU fallback is not built (${detail})`; }
      catch (error) {
        try { product?.destroy(); } catch { /* the device may be gone */ }
        product = null; this.warn = true;
        if (!rec.wasm) throw error;
        this.why = `FALLBACK to wasm: WebGPU could not run the generated program (${error.message || error})`;
      }
    } else if (rec.route === "webgpu-compute" || !rec.wasm) {
      if (!rec.wasm) throw new Error("needs WebGPU: this size has no wasm product");
      this.warn = true; this.why = `FALLBACK to wasm: the compiler chose WebGPU compute (${detail}) but navigator.gpu is absent here (needs https and a WebGPU browser)`;
    } else this.why = `compiler verdict: the dispatches do not pay (${detail})`;
    if (!product) { product = new WasmProduct(); await product.init(rec, this.manifest, Math.max(1, Math.min(16, Math.floor(spec.studio_count || 1)))); }
    this.product = product; this.productKey = key; this.settleStep = null; this.stepMs = 0;
    this.route = product.kind;
    // one snapshot per `snapEvery` steps: about one per displayed frame at x1, and a divisor of the settle rule's sample
    const set = this.manifest.settle, dt = this.manifest.law.dt;
    let snap = Math.max(1, Math.round(1 / (60 * dt)));
    while (set.sample % snap) snap--;
    this.tape = new Recording({ snapEvery: snap, frameBytes: product.frameBytes, capBytes: this.capBytes, keyEvery: snap * Math.max(1, Math.round(1 / (dt * snap))) });
    this.#recordCap();
    this.ready = true;
  }

  // RECORDING (readback saving) is OFF by default and switchable: off, the window holds only what the display needs (the
  // pump's lead plus a margin) and no keyframe state copies are read back, so Replay / Bounce / Continue are unavailable
  // (Rebirth needs nothing recorded).  On, the window grows to the memory cap and keyframes are kept again.
  setRecording(on) {
    this.recordOn = !!on;
    if (!this.tape) return;
    this.#recordCap();
    if (!this.recordOn) this.#release({ keys: this.tape.keys.splice(0) });
    this.onStatus();
  }

  #recordCap() {
    const t = this.tape;
    t.capFrames = this.recordOn ? Math.max(2, Math.floor(this.capBytes / t.frameBytes)) : LEAD_FRAMES + 4;
  }

  // give the recording's frames and keyframes back (frames to the product's pool, keyframes to the device)
  #release(gone) {
    if (!gone || !this.product) return;
    for (const f of gone.frames || []) if (f && f.length === this.product.frameBytes / 4 && this.product.pool.length < 64) this.product.pool.push(f);
    for (const k of gone.keys || []) if (k && k.snap !== undefined) this.product.freeKey(k.snap);
  }

  // the runtime inputs as the law should see them now: the server's defaults, replaced by the sensor's gravity (and the lid
  // and grid height that keep grains in the basin) while the page drives it
  #inputsNow() {
    const out = {};
    for (const name of this.runtimeInputs) if (name in this.spec.uniforms) out[name] = this.spec.uniforms[name];
    if (this.bodyTau != null && this.runtimeInputs.has("body_tau")) out.body_tau = this.bodyTau;   // the hinged body's coupling torque (setBodyTorque)
    if (this.sand && this.spec.purrline) {                // the purrline game's solids (setPurrline): the occupancy mask and the falling items
      const mine = this.runtimeInputs, w = purrlineInputs(this.sand, this.spec.purrline.map, this.spec.purrline.boxes);
      for (const name of Object.keys(w)) if (mine.has(name)) out[name] = w[name];
    }
    if (this.spec.mobroad) {                              // [MOBROAD] the road's bullets as the law's kill segments (setMobroad), swept from where they were last written
      const mine = this.runtimeInputs, w = mobroadInputs(this.mob || [], this.mobPrev, this.spec.mobroad);
      for (const name of Object.keys(w)) if (mine.has(name)) out[name] = w[name];
      const aim = { ...(this.mobAim ? mobroadTarget(this.mobAim.target[0], this.mobAim.target[1], this.spec.mobroad) : {}), ...(this.mobAim && this.mobAim.pather ? mobroadPather(this.mobAim.pather) : {}) };
      for (const name of Object.keys(aim)) if (mine.has(name)) out[name] = aim[name];   // the player's place (the pather's target) and the pather's weights (hand-set / trained)
    }
    const g = this.spin ? spinGravity(this.spin, this.steps * this.manifest.law.dt, this.spinMotion) : this.gravity;
    const flip = this.caseMotionAt(this.steps);
    if (this.runtimeInputs.has("case_omega")) {          // the case's own turn, from the same motion (one mechanism): 0 when it stands still
      const [om, dom] = this.spin ? spinOmega(this.spin, this.steps * this.manifest.law.dt, this.spinMotion) : [0, 0];
      const axis = this.caseAxis || "y";
      if (axis === "y") { out.case_omega = om + flip.omega; out.case_domega = dom + flip.domega; }
      else {
        const [x, y, z] = lawAxis(axis);
        out.case_omega_x = x * flip.omega; out.case_omega_y = y * flip.omega; out.case_omega = z * flip.omega;
        out.case_domega_x = x * flip.domega; out.case_domega_y = y * flip.domega; out.case_domega = z * flip.domega;
      }
    }
    if (["case_acc_x", "case_acc_y", "case_acc_z"].every((name) => this.runtimeInputs.has(name))) {
      // Uniforms are held for the next snapshot's steps; sample its midpoint.
      const a = this.caseDisplacementAt(this.steps + this.snap / 2).acceleration;
      out.case_acc_x = a[0]; out.case_acc_y = a[1]; out.case_acc_z = a[2];
    }
    if (g) {
      const G = this.manifest.law.gravity;
      out.grav_x = g[0] * G; out.grav_y = g[1] * G; if (this.runtimeInputs.has("grav_z")) out.grav_z = (g[2] || 0) * G;   // the 3-D law only; a spinning case turns in the screen's plane (no z)
      Object.assign(out, this.spec.tilt || {});
    }
    // THE CASE'S STANDING ANGLE (the page's "case angle" / "turn over"): the case is held turned by caseTilt radians, so gravity in its
    // frame is turned the same way (R(a): the convention of spinGravity, whose angle adds to this one)
    const a = flip.angle;
    if (a && "grav_x" in out && "grav_y" in out) {
      if ((this.caseAxis || "y") === "y") {
        const c = Math.cos(a), s = Math.sin(a), gx = out.grav_x, gy = out.grav_y;
        out.grav_x = gx * c - gy * s; out.grav_y = gx * s + gy * c;
      } else {
        const [gx, gy, gz] = rotateLawVector([out.grav_x, out.grav_y, out.grav_z || 0], this.caseAxis, a);
        out.grav_x = gx; out.grav_y = gy; out.grav_z = gz;
      }
    }
    // the gravity scale (a page slider): every source -- the default pull, the phone, the camera, a spinning case -- at k x
    const k = this.gravityScale ?? 1;
    if (k !== 1) for (const name of ["grav_x", "grav_y", "grav_z"]) if (name in out) out[name] *= k;
    for (const name of Object.keys(out)) out[name] = Math.fround(out[name]);
    return out;
  }

  // [GHOSTS] (replay_layers.js) a second runtime that re-runs `from`'s settings from the start as a background ghost run: the page's
  // gravity sources and case angle (plus `caseTiltAdd`, the seed's perturbation) and the image feeds are taken over; nothing is shared.
  inheritFrom(from, { caseTiltAdd = 0 } = {}) {
    this.gravity = from.gravity ? [...from.gravity] : null; this.spin = from.spin; this.spinMotion = from.spinMotion ? { ...from.spinMotion } : this.spinMotion;
    this.vibration = from.vibration ? { ...from.vibration } : null;
    this.nudge = from.nudge ? { ...from.nudge, direction: [...from.nudge.direction] } : null;
    this.caseTilt = (from.caseTilt || 0) + caseTiltAdd; this.gravityScale = from.gravityScale; this.bodyTau = from.bodyTau; this.named = from.named;
  }
  // [GHOSTS] the spec's uniforms with the nozzle speed scaled by 1 + rel (a runtime input; absent in a law without a nozzle)
  perturbedUniforms(uniforms, rel) {
    const out = { ...uniforms };
    if (rel && typeof out.nozzle_speed === "number") out.nozzle_speed = out.nozzle_speed * (1 + rel);
    return out;
  }
  // [GHOSTS] stop and free this runtime (a ghost run's, once its frames are taken)
  dispose() {
    this.running = false; this.spec = null; this.token++;
    if (this.tape) this.#release(this.tape.clear());
    if (this.product) { try { this.product.destroy(); } catch { /* the device may be gone */ } this.product = null; }
    this.ready = false;
  }

  setCaseTilt(radians) {
    const next = Number(radians) || 0;
    if (next === (this.caseTilt || 0) && !(this.caseFlips || []).length && (this.caseAxis || "y") === "y") return;
    this.caseTilt = next;
    this.caseFlips = [];
    this.caseAxis = "y";
    this.#kick();
  }

  caseMotionAt(step) {
    if (this.casePeriodic && this.manifest) {
      const p = this.casePeriodic, t = step * this.manifest.law.dt;
      const w = 2 * Math.PI * p.hz, phase = w * t + p.phase;
      if (p.kind === "rock") {
        const A = p.arc * Math.PI / 360;
        return { angle: A * Math.sin(phase), omega: -A * w * Math.cos(phase), domega: A * w * w * Math.sin(phase) };
      }
      return { angle: phase, omega: -w, domega: 0 };
    }
    const flips = this.caseFlips || [];
    let motion = { angle: this.caseTilt || 0, omega: 0, domega: 0 };
    for (const flip of flips) if (step >= flip.startStep) motion = caseFlipAt(flip, step); else break;
    return motion;
  }
  caseAngleAt(step) {
    const held = this.caseMotionAt(step).angle;
    return held + (this.spin && this.manifest && (this.caseAxis || "y") === "y" ? gravityAngle(this.spin, step * this.manifest.law.dt, this.spinMotion) : 0);
  }
  // Continuous prescribed case motion through the same angular inputs as
  // finite physical flips. The requested rate is bounded by contact capacity.
  setCasePeriodic(kind, hz = 0, arc = 60, phaseDegrees = 0, axis = "y") {
    if (!kind || kind === "off") { this.casePeriodic = null; this.#kick(); return { hz: 0, axis }; }
    if (!this.ready || !this.manifest || !["spin", "rock"].includes(kind) || !["x", "y", "z"].includes(axis)) return null;
    if (axis !== "y" && !["case_omega_x", "case_omega_y", "case_domega_x", "case_domega_y", "grav_z"].every((name) => this.runtimeInputs.has(name))) return null;
    const u = { ...this.rec.uniforms, ...this.uniforms, ...this.lawInputs };
    const cell = u.cell ?? 0.028 * u.cell_scale;
    const reach = Math.hypot(u.half, 0.5 * u.rows * cell, axis === "y" ? 0 : u.dh) + 1e-6;
    const radius = Math.min(...[0, 1, 2, 3].map((i) => u[`r${i}`]).filter((r) => r > 0));
    const maxOmega = 0.98 * u.omega * Math.sqrt(0.5 * radius / reach);
    const maxDomega = 0.98 * u.omega * u.omega * 0.5 * radius / reach;
    const A = kind === "rock" ? Math.max(1e-3, Math.PI * Number(arc) / 360) : 1;
    const limit = kind === "rock" ? Math.min(maxOmega / A, Math.sqrt(maxDomega / A)) : maxOmega;
    const effectiveHz = Math.max(0, Math.min(Number(hz) || 0, limit / (2 * Math.PI)));
    this.spin = null; this.caseTilt = 0; this.caseFlips = []; this.caseAxis = axis;
    this.casePeriodic = { kind, hz: effectiveHz, arc: Number(arc), phase: Number(phaseDegrees) * Math.PI / 180 };
    this.#kick();
    return { hz: effectiveHz, axis, limited: effectiveHz < Number(hz) - 1e-6 };
  }
  // Prescribed finite case turn about the selected world axis (2-D supports the case normal only).
  // The law limits angular terms by spring contact capacity per grain.  Choose a duration that stays below even the
  // smallest currently live grain's cap; report the actual duration to the caller.
  startCaseFlip(seconds = 0.4, direction = 1, axis = "y") {
    if (!this.ready || !this.tape || !this.manifest || !this.runtimeInputs.has("case_omega")) return null;
    if (!["x", "y", "z"].includes(axis) || (axis !== "y" && !["case_omega_x", "case_omega_y", "case_domega_x", "case_domega_y", "grav_z"].every((name) => this.runtimeInputs.has(name)))) return null;
    if (axis !== "y" && (this.spin || ((this.caseAxis || "y") === "y" && this.caseTilt))) return null;
    if ((this.caseFlips || []).length && axis !== (this.caseAxis || "y")) return null;
    const data = this.tape.get(this.tape.hi);
    if (!data) return null;
    let radius = Infinity;
    for (let i = 0; i < this.rec.n; i++) {
      const r = decodeMeta(data[5 * i + 4]).radius;
      if (r > 0) radius = Math.min(radius, r);
    }
    if (!Number.isFinite(radius)) return null;
    const u = { ...this.rec.uniforms, ...this.uniforms, ...this.lawInputs }, cell = u.cell ?? 0.028 * u.cell_scale;
    const reach = Math.hypot(u.half, 0.5 * u.rows * cell, axis === "y" ? 0 : u.dh) + 1e-6;
    const w = 2 * Math.PI * Math.abs(this.spin || 0), rock = this.spinMotion && this.spinMotion.kind === "rock";
    const amp = rock ? Math.PI * this.spinMotion.arc / 360 : 1;
    const maxOmega = 0.98 * u.omega * Math.sqrt(0.5 * radius / reach) - amp * w;
    const maxDomega = 0.98 * u.omega * u.omega * 0.5 * radius / reach - (rock ? amp * w * w : 0);
    if (!(maxOmega > 0 && maxDomega > 0)) return null;
    const delta = (direction < 0 ? -1 : 1) * Math.PI, dt = this.manifest.law.dt;
    const steps = caseFlipSteps(delta, Math.max(0.1, Number(seconds) || 0.4), dt, maxOmega, maxDomega);
    const flip = { from: this.caseMotionAt(this.steps).angle, delta, startStep: this.steps, steps, dt };
    this.caseAxis = axis;
    (this.caseFlips ||= []).push(flip);
    this.#kick();
    return { requestedSeconds: seconds, seconds: steps * dt, maxOmega, maxDomega, axis };
  }

  // The hinged body's (waterwheel) machine-sim coupling: the coupling writes its transmitted torque here (an absolute runtime input of the law,
  // logged with its step like the others) and reads the wheel's speed back from the law's diagnostics.
  setBodyTorque(nm) {
    const next = Number(nm) || 0;
    if (next === (this.bodyTau ?? 0)) return;
    this.bodyTau = next;
    this.#kick();
  }
  get bodySpeed() { return this.diag ? this.diag.body_om ?? null : null; }
  // The wheel is rendered at the playback cursor, so its angle must come from
  // the law snapshot displayed there rather than the latest status diagnostic.
  bodyAngle() {
    const name = this.spec?.body?.angle || "body_th";
    return (this.shown == null ? null : recordedAngle(this.tape, this.play.pos, name))
      ?? this.diag?.[name] ?? 0;
  }

  // PURRLINE SAND: the purrline game's state (``purrline`` of the game state, its falling list already at this instant), written to the law's
  // solids each batch -- ONE WAY: nothing the sand does reaches the game.  null stops it (the solids are then written empty).
  setPurrline(game) { this.sand = game || null; }

  // [MOBROAD] the road's bullets ([{id, x, y, kr}], board units): written to the law's kill segments each batch -- ONE WAY.  The law's own counts come
  // back through ``diag`` (n_dead, n_goal, n_crush).
  setMobroad(bullets) { this.mob = bullets || null; }
  // [MOBROAD] the player's board place and the pather's weights (null: the spec's), written into the law's inputs each batch (one way)
  setMobroadAim(target, pather) { this.mobAim = target ? { target, pather: pather || null } : null; }

  setGravityScale(k) {
    const next = Math.max(0, Number(k));
    if (next === (this.gravityScale ?? 1)) return;
    this.gravityScale = next;
    this.#kick();
  }

  // Write the runtime inputs to the device if they changed (or `force`), at this step boundary, and log them with the step.
  #applyInputs(force) {
    const now = this.#inputsNow(), changed = Object.keys(now).some((n) => now[n] !== this.lawInputs[n]);
    if (!changed && !force) return;
    this.lawInputs = now;
    if (this.spec.mobroad) this.mobPrev = new Map((this.mob || []).map((b) => [b.id, [b.x, b.y]]));   // [MOBROAD] the next segments start where these end
    for (const [name, value] of Object.entries(now)) this.product.setUniform(name, value);
    this.tape.setInputs(this.steps, now);
  }

  // The phone's gravity (or the off switch: null).  Fractions of g in the screen's plane; takes effect at the next batch.
  // A SPINNING CASE: gravity turns at ``hz`` revolutions per SIMULATED second (negative = clockwise), as if the box rotated
  // at a prescribed speed; 0 or null stops it.  The angle follows the law's own step count (not the wall clock), so a
  // free-running or replayed run turns the same way, and it overrides the phone's gravity while it runs.
  // ``motion``: "spin" (a full turn per cycle) or "rock" (gravity swings sinusoidally through ``arc`` degrees, centred on
  // straight down, one swing out and back per cycle) -- a scripted rocker until the teeter-totter case is a law body.
  setSpin(hz, motion = this.spinMotion || { kind: "spin", arc: 60 }) {
    const next = hz ? Number(hz) : null;
    this.spinMotion = { kind: ["rock", "flip"].includes(motion.kind) ? motion.kind : "spin", arc: Math.max(0, Math.min(360, Number(motion.arc) || 0)),
                        phase: Number(motion.phase) || 0 };
    this.spin = this.spinMotion.kind === "flip" ? null : next;
    this.#kick();
  }

  setCaseVibration(kind, hz, amplitude, axis = "z", phase = 0) {
    const supported = ["case_acc_x", "case_acc_y", "case_acc_z"].every((name) => this.runtimeInputs.has(name));
    const mode = ["sander", "vibrator"].includes(kind) ? kind : null;
    if (!supported && mode) { this.vibration = null; return null; }
    this.vibration = mode ? { kind: mode, hz: Math.min(8, Math.max(0.1, Number(hz) || 1)),
      amplitude: Math.min(0.04, Math.max(0, Number(amplitude) || 0)), axis: ["x", "y", "z"].includes(axis) ? axis : "z",
      phase: Number(phase) || 0 } : null;
    this.#kick();
    return this.vibration;
  }

  startCaseNudge(x, y) {
    if (!this.ready || !["case_acc_x", "case_acc_y"].every((name) => this.runtimeInputs.has(name))) return null;
    const magnitude = Math.hypot(x, y);
    if (!(magnitude >= 0.08)) return null;
    if (this.nudge && (this.shown == null || this.shown * this.snap < this.nudge.endStep)) return null;
    const duration = 0.22, startStep = this.steps;
    this.nudge = { direction: [x / magnitude, y / magnitude], strength: Math.min(1, magnitude), startStep,
      endStep: startStep + Math.ceil(duration / this.manifest.law.dt), duration };
    this.#kick();
    return this.nudge;
  }

  caseDisplacementAt(step) {
    const v = this.vibration && ["case_acc_x", "case_acc_y", "case_acc_z"].every((name) => this.runtimeInputs.has(name)) ? this.vibration : null;
    const a = v ? caseVibration(v.kind, v.hz, v.amplitude, v.axis, v.phase, step * (this.manifest?.law?.dt || 0))
      : { displacement: [0, 0, 0], acceleration: [0, 0, 0] };
    if (!this.nudge) return a;
    const b = caseNudge(this.nudge.direction, this.nudge.strength, this.nudge.startStep, step,
      this.manifest.law.dt, this.nudge.duration);
    return { displacement: a.displacement.map((value, i) => value + b.displacement[i]),
      acceleration: a.acceleration.map((value, i) => value + b.acceleration[i]) };
  }

  setGravity(gx, gy, gz = 0) {
    const next = gx == null ? null : [gx, gy, gz], old = this.gravity;
    if (next && old && Math.hypot(next[0] - old[0], next[1] - old[1], next[2] - old[2]) < 0.004) return;
    this.gravity = next;
    this.#kick();
  }

  #reset(phase2Rest) {
    this.caseTilt = this.caseMotionAt(this.steps).angle;
    this.nudge = null;
    this.caseFlips = [];
    this.token++; this.arrived.clear(); this.rotations.clear(); this.emojiTumble.clear();
    this.#release(this.tape.clear());
    this.lawInputs = this.#inputsNow();
    this.product.reset({ named: { ...this.named, ...this.start }, uniforms: { ...this.rec.uniforms, ...this.uniforms, ...this.lawInputs } });
    this.steps = 0; this.running = true; this.last = performance.now() / 1000; this.resimming = false;
    this.restPending = true; this.lastDone = 0; this.diagAt = 0;
    this.tape.setInputs(0, this.lawInputs);
    this.play.set("live", this.tape); this.play.pos = 0; this.shown = null;
    // the stop step S belongs to phase 1's run; phase 2 (the image replay) keeps it and stops at exactly S
    if (!this.uniforms.use_image) this.settleStep = null;
    this.detector = new SettleDetector(this.manifest.settle, this.rec.n, this.rec.pad); this.diag = null;
    if (!phase2Rest) this.note = "";
    this.#kick();
  }

  // --- image -> per-id feeds (a game's `file` control with a `sample` spec): the real file path ---
  async pickImage(file, spec) {
    if (!file) return;
    const [rw, rh] = spec.raster;
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const canvas = document.createElement("canvas");
    canvas.width = rw; canvas.height = rh;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, rw, rh);
    const scale = Math.max(rw / bitmap.width, rh / bitmap.height);
    ctx.drawImage(bitmap, (rw - bitmap.width * scale) / 2, (rh - bitmap.height * scale) / 2, bitmap.width * scale, bitmap.height * scale);
    if (bitmap.close) bitmap.close();
    this.setRaster({ w: rw, h: rh, data: ctx.getImageData(0, 0, rw, rh).data }, spec);
  }

  setRaster(raster, spec) {
    this.raster = raster; this.sampleSpec = spec; this.verdict = null;
    if (this.rest) this.#sample(this.rest);
    else { this.note = this.settleMode ? "image loaded: it is mapped when the pile has settled" : "image loaded: set run to 'stop at rest' to map it onto a settled pile"; this.onStatus(); }
  }

  // The image is ONLY sampled into the per-id feeds, at each id's ACTUAL rest position
  // in the recorded pile (normalised by the pile's own extent); nothing of it is drawn.
  #sample(rest) {
    const spec = this.sampleSpec, n = rest.x.length;
    const { bytes, channels } = sampleRasterAt(this.raster, rest.x, rest.y, n, spec.into);
    this.named = Object.fromEntries(spec.into.map((name, c) => [name, channels[c]]));
    this.note = "image mapped at the rest positions";
    const post = spec.post;
    if (n <= post.max_ids) {
      let text = "";
      for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      this.onSamples(post, { [post.field]: btoa(text) });
    } else this.onSamples(post, { count: n });      // too many to ship: the server only needs to know the image is mapped
  }

  // When the pile has settled (an event): phase 1 records its rest positions; phase 2 is checked against them.
  async #recordRest() {
    const phase2 = !!this.uniforms.use_image;
    const cols = await this.product.columnsNow(phase2 ? ["x", "y", "cr", "cg", "cb"] : ["x", "y"]);
    if (!phase2) {
      this.rest = { x: cols.x, y: cols.y }; this.verdict = null;
      if (this.raster) this.#sample(this.rest);
      this.note = this.raster ? this.note : "settled: rest positions recorded";
    } else if (this.rest) {
      const n = cols.x.length;
      let equal = 0, worst = 0, formed = 0;
      const named = this.named, [r, g, b] = this.sampleSpec.into;
      for (let i = 0; i < n; i++) {
        const same = cols.x[i] === this.rest.x[i] && cols.y[i] === this.rest.y[i];
        if (same) equal++; else worst = Math.max(worst, Math.abs(cols.x[i] - this.rest.x[i]), Math.abs(cols.y[i] - this.rest.y[i]));
        if (cols.cr[i] === named[r][i] && cols.cg[i] === named[g][i] && cols.cb[i] === named[b][i]) formed++;
      }
      const identical = equal === n;
      this.verdict = { identical, equal, n, worst, formed, recovered: this.recovering };
      if (!identical && !this.recovering) {
        // This device's replay is not bit-identical: record THIS run's rest positions, sample the image
        // there, and replay once more, so the image still forms from positions this device repeats.
        this.recovering = true;
        this.rest = { x: cols.x, y: cols.y };
        this.#sample(this.rest);
        this.note = `replay NOT bit-identical on this device (${n - equal}/${n} differ, worst ${worst.toExponential(2)}): positions recorded per device, replaying again`;
        this.#reset(true);
      } else if (!identical) this.note = "device replay still not bit-identical; the image is formed from this device's own recorded positions";
      else this.recovering = false;
    }
    this.onStatus();
  }

  // --- the pump: runs as fast as the device allows, apart from the display's frames ---
  #kick() {
    if (this.kicked || !this.ready || typeof MessageChannel === "undefined") return;
    this.kicked = true;
    if (!this.channel) { this.channel = new MessageChannel(); this.channel.port1.onmessage = () => { this.kicked = false; this.#tick(); }; }
    this.channel.port2.postMessage(0);
  }

  #stop() { return this.settleStep ?? Math.ceil(this.manifest.settle.max_seconds / this.manifest.law.dt - 1e-9); }
  get turbo() { return !!this.spec && this.spec.speed === 0; }
  get lead() { return this.steps / this.snap - Math.floor(this.play.pos); }

  #canSubmit() {
    const p = this.product, settle = this.settleMode;
    if (!this.ready || !p || !this.running || !this.spec || this.resimming || this.play.mode !== "live") return false;
    if (!p.canBatch() || this.inflight >= (settle ? 1 : MAX_IN_FLIGHT)) return false;      // settle reads each sample before going on, so it is exact
    if (settle && this.steps >= this.#stop()) return false;
    if (!this.turbo && this.lead >= LEAD_FRAMES) return false;
    return true;
  }

  #tick() {
    const p = this.product, t0 = performance.now(), budget = p && p.kind === "wasm" ? 8 : 3;
    while (this.#canSubmit()) {
      try { this.#submit(); } catch (error) { this.error = String((error && error.message) || error); this.running = false; this.onStatus(); return; }
      if (performance.now() - t0 > budget) { this.#kick(); return; }
    }
  }

  // how many snapshots the next batch holds: what fits the target time at the measured step cost, the buffer's room, and in
  // settle mode exactly up to the next sample (or the stop step)
  #batchFrames() {
    const p = this.product, snap = this.snap, set = this.manifest.settle;
    let m = this.stepMs ? Math.floor((p.kind === "wasm" ? TARGET_MS.wasm : TARGET_MS.gpu) / (this.stepMs * snap)) : 1;
    m = Math.max(1, Math.min(m, p.maxFrames));
    if (!this.turbo) m = Math.max(1, Math.min(m, LEAD_FRAMES - Math.floor(this.lead)));
    if (this.settleMode) m = Math.max(1, Math.min((set.sample - (this.steps % set.sample)) / snap, (this.#stop() - this.steps) / snap));
    // a falling item moves by position, once per batch: with items in the air a batch is one snapshot (6 steps), so a box is never more than that stale
    if (this.sand && this.sand.falling && this.sand.falling.length) m = 1;
    // the inputs change only between batches: a spinning case keeps each batch to SPIN_TURN of a revolution
    if (this.spin) {
      // the fastest the gravity angle turns, in revolutions per simulated second: |hz| for a spin, pi * arc/360 * |hz| at the
      // middle of a rock's swing
      const rate = Math.abs(this.spin) * (this.spinMotion && this.spinMotion.kind === "rock" ? Math.PI * this.spinMotion.arc / 360 : 1);
      if (rate > 0) m = Math.max(1, Math.min(m, Math.floor(SPIN_TURN / (rate * this.manifest.law.dt * snap))));
    }
    if (this.vibration) m = Math.max(1, Math.min(m, Math.floor(1 / (12 * this.vibration.hz * this.manifest.law.dt * snap))));
    if (this.nudge && this.steps < this.nudge.endStep) m = 1;
    if ((this.caseFlips || []).some((f) => this.steps >= f.startStep && this.steps < f.startStep + f.steps)) m = 1;
    return m;
  }

  #submit() {
    const p = this.product, snap = this.snap, set = this.manifest.settle, settle = this.settleMode, keyEvery = this.tape.keyEvery;
    const m = this.#batchFrames(), base = this.steps, first = base / snap + 1, diagAt = [], keyAt = [], now = performance.now();
    this.#applyInputs(false);
    for (let j = 0; j < m; j++) {
      const at = base + (j + 1) * snap;
      if (this.spec.body?.angle || (settle ? at % set.sample === 0 : (j === m - 1 && now - this.diagAt > DIAG_MS))) diagAt.push(j);
      if (this.recordOn && at % keyEvery === 0) keyAt.push(j);
    }
    if (diagAt.length && !settle) this.diagAt = now;
    this.steps += m * snap; this.inflight++;
    const token = this.token;
    p.batch({ frames: m, snapEvery: snap, diagAt, keyAt, emojiSpin: this.emojiTumbleOn }).then(
      (result) => { this.inflight--; this.#land(token, first, m, result, now); },
      (error) => { this.inflight--; this.error = String((error && error.message) || error); this.running = false; this.onStatus(); });
  }

  #land(token, first, m, result, started) {
    const done = performance.now(), snap = this.snap;
    if (token !== this.token) { this.#release({ frames: result.frames, keys: result.keys.filter(Boolean).map((k) => ({ snap: k.payload })) }); this.#kick(); return; }
    const per = (done - Math.max(started, this.lastDone)) / (m * snap);
    this.lastDone = done; this.stepMs = this.stepMs ? this.stepMs * 0.7 + per * 0.3 : per;
    this.stepsDone += m * snap;
    this.arrived.set(first, { m, result });
    while (this.arrived.has(this.tape.hi + 1)) {
      const k0 = this.tape.hi + 1, { m: count, result: r } = this.arrived.get(k0);
      this.arrived.delete(k0);
      for (let j = 0; j < count; j++) {
        const k = k0 + j, step = k * snap;
        if (this.onSnapshot) this.onSnapshot(k, r.frames[j]);
        this.#release(this.tape.push(k, r.frames[j], r.diags[j]));
        const omegas = r.spins?.[j];
        if (omegas) {
          const prior = this.rotations.get(k - 1);
          this.rotations.set(k, omegas.map((omega, id) => {
            if (!omega.every(Number.isFinite)) return null;
            const previous = prior?.[id];
            const mean = previous ? omega.map((v, i) => (v + previous.omega[i]) / 2) : omega;
            const quaternion = integrateAngular(previous?.quaternion || [0, 0, 0, 1], mean, snap * this.manifest.law.dt);
            return { frame: k, quaternion, omega };
          }));
        }
        while (this.rotations.size && this.rotations.keys().next().value < this.tape.lo)
          this.rotations.delete(this.rotations.keys().next().value);
        if (r.emojiSpins?.[j]) this.emojiTumble.add(k, r.emojiSpins[j], snap * this.manifest.law.dt, this.tape.lo);
        if (r.keys[j]) this.#release({ keys: this.tape.addKey(step, r.keys[j].payload, r.keys[j].bytes) });
        if (r.diags[j]) this.#takeDiag(step, r.diags[j]);
      }
    }
    if (this.settleMode && this.restPending && this.steps >= this.#stop() && this.inflight === 0) { this.restPending = false; this.#recordRest(); }
    this.#kick();
  }

  // the law's diagnostics after `step` steps; in settle mode they feed the settle rule (which stops the run)
  #takeDiag(step, d) {
    this.diag = { step, ...d };
    if (!this.settleMode || step % this.manifest.settle.sample !== 0) return;
    const found = this.detector.push(step, d);
    if (found !== null && this.settleStep === null) this.settleStep = found;
    if (step >= Math.ceil(this.manifest.settle.max_seconds / this.manifest.law.dt - 1e-9) && this.settleStep === null) this.settleStep = step;
  }

  // --- per displayed frame: move the cursor through the recording ---
  frame(now) {
    if (!this.ready || !this.spec || !this.tape) return;
    const elapsed = Math.min(0.1, Math.max(0, now - this.last));
    this.last = now;
    this.frames++;
    if (now - this.fpsAt > 0.5) {
      this.stepsPerSec = this.stepsDone / (now - this.fpsAt); this.stepsDone = 0;
      this.fps = this.frames / (now - this.fpsAt); this.frames = 0; this.fpsAt = now; this.onStatus();
    }
    const speed = this.spec.speed ?? 1, dt = this.manifest.law.dt;
    this.shown = this.play.advance(this.turbo ? 0 : elapsed * speed / (this.snap * dt), this.tape, { turbo: this.turbo, maxLead: 4 * LEAD_FRAMES });
    if (this.play.mode === "live") this.#kick();      // the display used up some of the buffer
  }

  // the packed points to draw now (the cursor's frame of the recording)
  points() { return this.ready && this.tape && this.shown != null ? this.tape.get(this.shown) : null; }
  rotationFor(id) { return this.shown == null ? null : this.rotations.get(this.shown)?.[id] || null; }
  rotation() { return this.rotationFor(0); }
  setEmojiTumble(on) { this.emojiTumbleOn = !!on; }
  emojiPose() { return this.shown == null ? null : this.emojiTumble.get(this.shown); }

  // --- the run modes: the page's controls call compute.control(name) ---
  control(name) {
    if (!this.ready || !this.tape) return;
    if (name === "rebirth") this.#reset();
    else if (name === "replay" || name === "bounce") {
      if (this.tape.hi < this.tape.lo) return;
      this.play.set(name, this.tape); this.shown = Math.floor(this.play.pos);
    } else if (name === "continue") this.#continue();
  }

  // Resume the simulation from the displayed frame.  The recording after it is dropped (the run branches there) and the law's
  // state at that frame is rebuilt exactly: the nearest keyframe, then the logged steps and inputs run again.
  async #continue() {
    if (this.play.mode === "live" || this.resimming) return;
    this.resimming = true;
    const f = Math.floor(this.play.pos), token = this.token;
    while (this.inflight > 0) await new Promise((resolve) => setTimeout(resolve, 2));
    if (token !== this.token) return;
    const tape = this.tape, p = this.product;
    if (f < tape.hi) {
      const plan = tape.plan(f);
      if (plan.key) p.restore(plan.key.snap);
      else p.reset({ named: { ...this.named, ...this.start }, uniforms: { ...this.rec.uniforms, ...this.uniforms, ...(plan.initial || {}) } });
      for (const [name, value] of Object.entries(plan.initial || {})) p.setUniform(name, value);
      for (const op of plan.ops) {
        if (op.set) for (const [name, value] of Object.entries(op.set)) p.setUniform(name, value);
        else p.step(op.steps);
      }
      this.#release(tape.truncateAfter(f));
      for (const frame of this.rotations.keys()) if (frame > f) this.rotations.delete(frame);
      this.emojiTumble.truncateAfter(f);
      this.arrived.clear();
      this.steps = f * this.snap;
      await p.idle();
      if (token !== this.token) return;
    }
    this.#applyInputs(true);
    this.play.set("live", tape); this.play.jump(f); this.shown = f;
    this.resimming = false; this.#kick();
  }

  // what the page's controls show about the run modes
  localState() {
    return { tilt: this.gravity !== null, mode: this.play.mode, frames: this.tape ? this.tape.length : 0, ready: this.ready, recording: !!this.recordOn };
  }

  energyText() {
    const d = this.diag;
    if (!d) return "";
    const n = this.rec.pad, total = d.n_nan + d.n_out + d.n_parked + d.n_live;
    const bad = [d.n_nan ? `${d.n_nan} NaN` : "", d.n_out ? `${d.n_out} outside` : "", d.n_drop ? `${d.n_drop} contacts dropped` : "", total !== n ? `COUNT ${total}/${n}` : ""].filter(Boolean);
    const kT = d.ke / (((this.manifest.law.dof || 6) / 2) * Math.max(d.n_live, 1)), set = this.lawInputs.temp ?? this.spec.uniforms.temp ?? 0;       // ke = dof/2 n kT: kT/2 per degree of freedom (3-D: six; the 2-D disk: x, y, spin)
    return ` | E ${d.ke.toExponential(2)} T ${kT.toExponential(2)} (set ${set.toExponential(1)}) v ${Math.sqrt(d.vmax2).toFixed(2)} live ${d.n_live.toLocaleString()}/${this.rec.n.toLocaleString()}${bad.length ? " | " + bad.join(", ") : ""}`;
  }

  verdictText() {
    const v = this.verdict;
    if (!v) return "";
    const formedText = `image formed ${v.formed.toLocaleString()}/${v.n.toLocaleString()}`;
    return v.identical ? ` | replay identical ${v.equal.toLocaleString()}/${v.n.toLocaleString()}; ${formedText}`
                       : ` | replay NOT identical ${v.equal.toLocaleString()}/${v.n.toLocaleString()} (worst ${v.worst.toExponential(2)}); ${formedText}`;
  }

  statusText() {
    if (!this.ready) return this.error ? "compute: " + this.error : "compute: loading";
    const adapter = this.product.adapterInfo ? ` [${this.product.adapterInfo}]` : "";
    const dt = this.manifest.law.dt, snap = this.snap, t = (this.shown ?? 0) * snap * dt;
    const run = this.settleMode ? (this.settleStep !== null && this.steps >= this.settleStep ? `settled at step ${this.settleStep}` : "settling") : "continuous";
    const fail = this.product.failure ? ` | GPU error: ${this.product.failure}` : "";
    const rate = this.stepsPerSec ? ` | ${Math.round(this.stepsPerSec).toLocaleString()} steps/s (x${(this.stepsPerSec * dt).toFixed(1)} real time)` : "";
    const cost = this.stepMs ? ` | ${this.stepMs.toFixed(2)} ms/step` : "";
    const tape = this.tape, mode = this.play.mode;
    const rec = tape && tape.length ? ` | ${mode === "live" ? "rec" : mode} ${tape.lo}-${tape.hi} (${mb(tape.bytes)} of ${mb(this.capBytes)})` : "";
    const grav = this.gravity ? ` | tilt gravity (${this.gravity[0].toFixed(2)}, ${this.gravity[1].toFixed(2)}, ${this.gravity[2].toFixed(2)})` : "";
    return `${this.route}${adapter}: ${this.why} | t ${t.toFixed(1)} s ${run}${this.energyText()} | ${this.fps.toFixed(0)} fps${rate}${cost}${rec}${grav}${this.note ? " | " + this.note : ""}${this.verdictText()}${fail}`;
  }

  // test/diagnostics only: step the product directly (not recorded; use while the pump rests)
  advance(count) {
    const total = this.settleStep ?? Math.ceil(this.manifest.settle.max_seconds / this.manifest.law.dt - 1e-9);
    count = Math.min(count, total - this.steps);
    this.product.step(count); this.steps += count; return this.steps;
  }
  columnsNow(names) { return this.product.columnsNow(names); }
  writeColumn(name, values) { return !!(this.product && this.product.writeColumn && this.product.writeColumn(name, values)); }    // [LIVEVIEW] (the WebGPU product only)
}

// --------------------------------------------------- GL points (shared scene)
// Each particle is drawn at its TRUE radius in screen space: radius * (board scale) * focal / clip-w, from its own
// projection (so zoom, perspective and viewport size all enter exactly, with no minimum and no stand-in size).
//   * wider than ~one pixel: a disc, its edge antialiased by exact coverage;
//   * narrower: its AREA (pi r^2 px^2, times the display stride so the omitted ids are accounted for) is splatted with bilinear
//     sub-pixel weights, additively, as (colour * coverage, coverage) into a float target; a resolve pass over the board's
//     footprint divides colour by coverage and composites over the scene with alpha = min(coverage, 1).  Dense regions show
//     their true average colour; sparse ones stay faint.
const POINT_VS = `#version 300 es
${GRAIN_COLOR_GLSL}
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform mat4 uViewProj; uniform vec4 uMap; uniform vec3 uBoard; uniform float uFocal; uniform vec2 uViewport; uniform float uMax; uniform float uStride;
uniform int uPass;            // 0 splat (small), 1 disc (large), 2 small directly into the scene (no float target)
uniform float uThreshold;     // the diameter (px) below which a particle is accumulated
out vec4 vColor; out float vDiameter; out float vCover; flat out vec2 vCenter;
void main() {
  vColor = grainColor(aColor, aMeta); vDiameter = 0.0; vCover = 0.0; vCenter = vec2(0.0);
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
  if (aMeta < 50.0) return;
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  // world -> board with ONE scale (uMap: centre x, centre y, scale k): circles stay circles at every size
  vec4 clip = uViewProj * vec4(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.z + aPos.z * uMap.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z, 1.0);   // the slab's depth axis stands off the board toward the camera
  if (clip.w <= 0.0) return;
  float rpx = radius * uMap.z * uFocal / clip.w;              // screen radius in pixels, from the particle's own projection
  float d = 2.0 * rpx;
  bool small = d < uThreshold;
  if (uPass == 1) {
    if (small) return;
    gl_Position = clip; gl_PointSize = min(d + 2.0, uMax);   // +2: room for the antialiased rim
    vDiameter = d;
  } else {
    if (!small) return;
    gl_Position = clip;
    vCenter = (clip.xy / clip.w * 0.5 + 0.5) * uViewport;
    vCover = 3.14159265 * rpx * rpx * uStride;
    gl_PointSize = 2.0;
  }
}`;
const POINT_FS = `#version 300 es
precision highp float;
precision highp int;
in vec4 vColor; in float vDiameter; in float vCover; flat in vec2 vCenter; out vec4 FragColor;
uniform vec3 u_background;   // the page's shared background: what an unfilled (alpha < 1) particle fades to
uniform int uPass;
void main() {
  vec3 rgb = mix(u_background, vColor.rgb, vColor.a);
  if (uPass == 1) {
    float edge = clamp(0.5 * vDiameter - length(gl_PointCoord - 0.5) * (vDiameter + 2.0) + 0.5, 0.0, 1.0);
    if (edge <= 0.0) discard;
    FragColor = vec4(rgb, edge);
    return;
  }
  vec2 off = abs(gl_FragCoord.xy - vCenter);
  float w = max(0.0, 1.0 - off.x) * max(0.0, 1.0 - off.y);
  if (uPass == 2) FragColor = vec4(rgb, min(vCover * w, 1.0));
  else FragColor = vec4(rgb * vCover * w, vCover * w);
}`;
const RESOLVE_VS = `#version 300 es
uniform mat4 uViewProj; uniform vec3 uBoard; uniform float uMargin;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vec2 u = (c * 2.0 - 1.0) * (0.5 + uMargin);                 // the board's plane, with a margin for particles beyond its edge
  gl_Position = uViewProj * vec4(uBoard.x * (0.5 + u.x), uBoard.z, uBoard.y * (0.5 + u.y), 1.0);
}`;
const RESOLVE_FS = `#version 300 es
precision highp float;
uniform sampler2D uAccum; out vec4 FragColor;
void main() {
  vec4 a = texelFetch(uAccum, ivec2(gl_FragCoord.xy), 0);
  if (a.a <= 1e-4) discard;
  FragColor = vec4(a.rgb / a.a, min(a.a, 1.0));
}`;

const OUTLINE_VS = `#version 300 es
layout(location=0) in vec3 aP;
uniform mat4 uViewProj; uniform vec4 uMap; uniform vec3 uBoard;
void main() {
  gl_Position = uViewProj * vec4(uBoard.x * 0.5 + (aP.x - uMap.x) * uMap.z, uBoard.z + aP.z * uMap.z, uBoard.y * 0.5 - (aP.y - uMap.y) * uMap.z, 1.0);
}`;
const OUTLINE_FS = `#version 300 es
precision highp float;
uniform vec4 uColor; out vec4 FragColor;
void main() { FragColor = uColor; }`;

export class PointCloud {
  constructor(gl, link, compile) {
    this.gl = gl; this.link = link; this.compile = compile; this.outlineProgram = null;
    this.styles = new ParticleStyles(gl, link, compile);       // the selectable looks (particle_styles.js); `style` of draw picks one
    this.noise = new NoiseFilter(gl, link, compile);            // the render-only spectral noise filter layered over any look (noise_filter.js); `noise` of draw
    this.program = link(compile(gl.VERTEX_SHADER, POINT_VS), compile(gl.FRAGMENT_SHADER, POINT_FS));
    this.resolve = link(compile(gl.VERTEX_SHADER, RESOLVE_VS), compile(gl.FRAGMENT_SHADER, RESOLVE_FS));
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.dyn = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, PT_BYTES, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, PT_BYTES, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, PT_BYTES, 16);
    gl.bindVertexArray(null);
    this.emptyVao = gl.createVertexArray();
    this.max = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.loc = Object.fromEntries(["uViewProj", "uMap", "uBoard", "uFocal", "uViewport", "uMax", "uStride", "uPass", "uThreshold", "u_background", "uTint", "uGrain", "uLut", "uBands"].map((n) => [n, gl.getUniformLocation(this.program, n)]));
    this.rloc = Object.fromEntries(["uViewProj", "uBoard", "uMargin", "uAccum"].map((n) => [n, gl.getUniformLocation(this.resolve, n)]));
    // a float target to accumulate into; without one, small particles fall back to alpha-blended sub-pixel dots (same area weighting)
    this.floatOk = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
    this.target = null;
  }

  #target(w, h) {
    const gl = this.gl;
    if (this.target && this.target.w === w && this.target.h === h) return this.target;
    if (this.target) { gl.deleteTexture(this.target.tex); gl.deleteFramebuffer(this.target.fbo); this.target = null; }
    // Texture bindings are shared with the scene's passes: bind on a unit of our own, then give back what was there.
    const unit = gl.getParameter(gl.ACTIVE_TEXTURE), held = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindTexture(gl.TEXTURE_2D, held); gl.activeTexture(unit);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { this.floatOk = false; gl.deleteTexture(tex); gl.deleteFramebuffer(fbo); return null; }
    return (this.target = { w, h, tex, fbo });
  }

  // The chamber's walls as a wireframe: the basin outline (a polyline of vertices [x, y]) at the front and back plexiglass
  // (z = +-dh) and a rib at each vertex, in the same board mapping as the points.  Render only: the law's walls are the
  // contacts; this is what the grains are held by, drawn so the shape is visible (a funnel looks like a funnel).
  drawOutline(outline, dh, { viewProj, view, board, y }, color = [0.85, 0.9, 1.0, 0.55]) {
    const gl = this.gl;
    if (!outline || outline.length < 2) return;
    if (!this.outlineProgram) {
      this.outlineProgram = this.link(this.compile(gl.VERTEX_SHADER, OUTLINE_VS), this.compile(gl.FRAGMENT_SHADER, OUTLINE_FS));
      this.outlineVao = gl.createVertexArray(); this.outlineBuf = gl.createBuffer();
      gl.bindVertexArray(this.outlineVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.outlineBuf);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0); gl.bindVertexArray(null);
      this.outlineLoc = Object.fromEntries(["uViewProj", "uMap", "uBoard", "uColor"].map((n) => [n, gl.getUniformLocation(this.outlineProgram, n)]));
    }
    const lines = [];
    for (let i = 0; i + 1 < outline.length; i++) for (const z of [dh, -dh]) lines.push(outline[i][0], outline[i][1], z, outline[i + 1][0], outline[i + 1][1], z);
    for (const [px, py] of outline) lines.push(px, py, dh, px, py, -dh);
    const [x0, x1, y0, y1] = view, k = Math.min(board[0] / (x1 - x0), board[1] / (y1 - y0));
    gl.useProgram(this.outlineProgram);
    gl.uniformMatrix4fv(this.outlineLoc.uViewProj, false, viewProj); gl.uniform4f(this.outlineLoc.uMap, (x0 + x1) / 2, (y0 + y1) / 2, k, 0);
    gl.uniform3f(this.outlineLoc.uBoard, board[0], board[1], y); gl.uniform4fv(this.outlineLoc.uColor, color);
    gl.bindVertexArray(this.outlineVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.outlineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    gl.drawArrays(gl.LINES, 0, lines.length / 3);
    gl.depthMask(true); gl.disable(gl.BLEND); gl.depthFunc(gl.LESS); gl.bindVertexArray(null);
  }

  // data: packed points ([x, y, z, rgba8, meta] per displayed id); view: the size's world window; focal: the projection's
  // vertical scale (proj[5]), turned into pixels with the CURRENT viewport's height
  draw(data, opts) {
    this.#drawField(data, opts);
    if (opts.noise && data) this.noise.apply(data, { vao: this.vao, viewProj: opts.viewProj, view: opts.view, board: opts.board, y: opts.y, tint: opts.tint }, opts.noise);
    else this.noise.fallback = "";
  }
  #drawField(data, { viewProj, view, board, y, focal, stride, background, style, lighting, camera, tint, colorMode, grain, scene, heat, spinPose }) {
    const gl = this.gl;
    if (!data) return;
    const [x0, x1, y0, y1] = view, k = Math.min(board[0] / (x1 - x0), board[1] / (y1 - y0));
    const vp = gl.getParameter(gl.VIEWPORT), bound = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const W = vp[2], H = vp[3];
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    const n = data.length / PT_WORDS;
    // a selected style (particle_styles.js) draws the frame itself; "points", and a style this device cannot run, go on below
    if (style && style !== "points" && this.styles.draw(style, { data, n, vao: this.vao, buffer: this.dyn, tint, colorMode, grain, scene, heat, spinPose, viewProj, board, y, focal, stride, background, lighting, camera, W, H, vp, bound,
                                                                 map: [(x0 + x1) / 2, (y0 + y1) / 2, k], now: performance.now() })) return;
    const setUniforms = (pass) => {
      gl.uniformMatrix4fv(this.loc.uViewProj, false, viewProj);
      gl.uniform4f(this.loc.uMap, (x0 + x1) / 2, (y0 + y1) / 2, k, 0);
      gl.uniform3f(this.loc.uBoard, board[0], board[1], y);
      gl.uniform1f(this.loc.uFocal, focal * H / 2); gl.uniform2f(this.loc.uViewport, W, H); gl.uniform1f(this.loc.uMax, this.max);
      gl.uniform1f(this.loc.uStride, stride || 1); gl.uniform1f(this.loc.uThreshold, 1.25);
      gl.uniform3fv(this.loc.u_background, new Float32Array(background || [0.085, 0.085, 0.1]));
      grainUniforms(gl, (n) => this.loc[n], tint, grain);
      gl.uniform1i(this.loc.uPass, pass);
    };
    gl.useProgram(this.program);
    gl.depthMask(false);
    const target = this.floatOk ? this.#target(W, H) : null;
    if (target) {
      // 1. splat the sub-pixel particles' (colour * coverage, coverage) additively
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo); gl.viewport(0, 0, W, H);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      setUniforms(0);
      gl.drawArrays(gl.POINTS, 0, n);
      gl.bindFramebuffer(gl.FRAMEBUFFER, bound); gl.viewport(vp[0], vp[1], W, H);
      // 2. resolve over the board's footprint (depth-tested against the scene like the points always were)
      gl.bindVertexArray(this.emptyVao);
      gl.useProgram(this.resolve);
      gl.uniformMatrix4fv(this.rloc.uViewProj, false, viewProj); gl.uniform3f(this.rloc.uBoard, board[0], board[1], y); gl.uniform1f(this.rloc.uMargin, 2.0);
      // The resolve reads the accumulation target on unit 0 -- the unit of the scene's pbr material table -- so
      // put that binding back afterwards: left bound, the next frame's lit pass read every material (albedo,
      // opacity, emission) out of this float texture and the plate, the sun and the bulbs rendered black.
      const activeUnit = gl.getParameter(gl.ACTIVE_TEXTURE);
      gl.activeTexture(gl.TEXTURE0); const held = gl.getParameter(gl.TEXTURE_BINDING_2D);
      gl.bindTexture(gl.TEXTURE_2D, target.tex); gl.uniform1i(this.rloc.uAccum, 0);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindTexture(gl.TEXTURE_2D, held); gl.activeTexture(activeUnit);
      gl.bindVertexArray(this.vao); gl.useProgram(this.program);
    } else {
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      setUniforms(2);
      gl.drawArrays(gl.POINTS, 0, n);
    }
    // 3. the particles wider than a pixel: discs at their true radius
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    setUniforms(1);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.disable(gl.BLEND); gl.depthFunc(gl.LESS); gl.depthMask(true); gl.bindVertexArray(null);
  }
}
