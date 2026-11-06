// The page's generic COMPUTE-PRODUCT runtime.
//
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
// small generic compute pass into 16 bytes per DISPLAYED id -- [x, y, rgba8
// (as bits), meta] -- every `display_stride`-th id when the count is large,
// and that packed buffer alone is copied back (triple buffered, one or two
// frames late) for WebGL to draw as GL points in the shared scene.  No WGSL
// is read by a vertex stage, so devices with no vertex-stage storage buffers
// (Apple's) are not a case.  The wasm route packs the same layout on the CPU.
//
// Determinism: reset writes the same bytes every time and the law has no
// atomics and no scatter, so a replay from the same initial state on the same
// device repeats exactly.  The runtime records the rest positions when phase
// 1 settles (one readback, an event), compares phase 2's with them, and says
// so on screen; if a device is not bit-identical it records phase 2's own
// positions and replays again, so the image still forms.
import { TuringGraphProgram } from "./turing-wasm-shell.js";

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

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
@group(0) @binding(__PTSB__) var<storage, read_write> pts: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
  let j = g.x + g.y * grid.x * 64u;
  if (j >= __NDISP__u) { return; }
  let i = j * __STRIDE__u;
  let live = __AGE__ >= f32(i) * __RATE__f;
  let vx = __VX__;
  let vy = __VY__;
  let rest = (vx * vx + vy * vy) < __REST2__;
  let c = pack4x8unorm(vec4<f32>(__R__, __G__, __B__, 1.0));
  pts[4u * j] = __X__;
  pts[4u * j + 1u] = __Y__;
  pts[4u * j + 2u] = bitcast<f32>(c);
  pts[4u * j + 3u] = select(0.0, select(100.0, 200.0, rest) + __RAD__ * 1000.0, live);
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
    const bindingBytes = abi.slots_per_binding * rec.pad * 4;
    if (bindingBytes > L.maxStorageBufferBindingSize) return { ok: false, reason: `a ${mb(bindingBytes)} binding exceeds this device's maxStorageBufferBindingSize (${mb(L.maxStorageBufferBindingSize)})` };
    if (bindingBytes > L.maxBufferSize) return { ok: false, reason: `a ${mb(bindingBytes)} buffer exceeds this device's maxBufferSize (${mb(L.maxBufferSize)})` };
    if (abi.bindings + 1 > L.maxStorageBuffersPerShaderStage) return { ok: false, reason: `needs ${abi.bindings + 1} storage bindings per stage; the device allows ${L.maxStorageBuffersPerShaderStage}` };
    if (Math.max(abi.groups[0], abi.groups[1]) > L.maxComputeWorkgroupsPerDimension) return { ok: false, reason: `needs ${abi.groups[0]}x${abi.groups[1]} workgroups; the device allows ${L.maxComputeWorkgroupsPerDimension} per dimension` };
    return { ok: true, adapter, limits: L, bindingBytes };
  }

  async init(rec, manifest) {
    const check = await GpuProduct.limits(rec);
    if (!check.ok) throw new Error("refused: " + check.reason);
    const adapter = check.adapter, A = rec.wgsl_abi, B = A.bindings;
    this.device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: Math.max(check.bindingBytes, 134217728),
      maxBufferSize: Math.max(check.bindingBytes, 268435456),
      maxStorageBuffersPerShaderStage: Math.max(B + 1, 8) } });
    this.device.addEventListener?.("uncapturederror", (e) => { this.failure = String((e.error && e.error.message) || e.error); });
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
      .replaceAll("__NDISP__", String(ndisp)).replaceAll("__STRIDE__", String(stride)).replaceAll("__RATE__", rec.emit_rate.toPrecision(9))
      .replaceAll("__AGE__", at(slot(p.live[0]))).replaceAll("__X__", at(slot(p.x))).replaceAll("__Y__", at(slot(p.y)))
      .replaceAll("__VX__", at(slot(p.vx))).replaceAll("__VY__", at(slot(p.vy)))
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
    this.pts = this.device.createBuffer({ size: ndisp * 16, usage: S });
    this.staging = [0, 1, 2].map(() => ({ buffer: this.device.createBuffer({ size: ndisp * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
    const oom = await this.device.popErrorScope();
    if (oom) throw new Error(`out of memory: the device could not allocate ${mb(rec.memory.arena_bytes + 4 * ndisp * 16)} for ${rec.n.toLocaleString()} ids (${oom.message})`);
    this.group = this.device.createBindGroup({ layout, entries: [...this.lanes.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: B, resource: { buffer: this.head } }] });
    this.packGroup = this.device.createBindGroup({ layout: packLayout, entries: [...this.lanes.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: B, resource: { buffer: this.pts } }] });
    this.latest = null; this.stepMs = 0; this.pending = 0;
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
    this.latest = null;
  }

  setUniform(name, value) { this.device.queue.writeBuffer(this.head, this.abi.uniform_index[name] * 4, new Float32Array([value])); }

  setColours(named) {
    for (const [name, values] of Object.entries(named)) {
      const slot = this.abi.feed_slots[name];
      if (slot !== undefined) this.device.queue.writeBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot), values);
    }
  }

  step(count) {
    const encoder = this.device.createCommandEncoder(), g = this.abi.groups;
    for (let k = 0; k < count; k++) {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, this.group);
      for (const pipeline of this.pipelines) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(g[0], g[1], 1); }
      pass.end();
    }
    const pack = encoder.beginComputePass();
    pack.setPipeline(this.pack); pack.setBindGroup(0, this.packGroup);
    const tg = Math.ceil(this.ndisp / 64), gx = Math.min(tg, 16384);
    pack.dispatchWorkgroups(gx, Math.ceil(tg / gx), 1);
    pack.end();
    // back-pressure and a measured cost: the frame loop does not submit more while the last
    // submission is unfinished, and the time to finish it, per step, is the step cost
    const submitted = performance.now();
    this.pending++;
    const settle = () => {
      this.pending--;
      if (count > 0) { const per = (performance.now() - submitted) / count; this.stepMs = this.stepMs ? this.stepMs * 0.8 + per * 0.2 : per; }
    };
    const free = this.staging.find((s) => !s.busy);
    if (free) {
      free.busy = true;
      encoder.copyBufferToBuffer(this.pts, 0, free.buffer, 0, this.ndisp * 16);
      this.device.queue.submit([encoder.finish()]);
      free.buffer.mapAsync(GPUMapMode.READ).then(() => {
        this.latest = new Float32Array(free.buffer.getMappedRange().slice(0));
        free.buffer.unmap(); free.busy = false;
      }).catch(() => { free.busy = false; });
    } else this.device.queue.submit([encoder.finish()]);
    this.device.queue.onSubmittedWorkDone().then(settle);
  }

  get busy() { return this.pending > 0; }
  points() { return this.latest; }

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

  nudge(ids, dx) {              // test hook: perturb a few ids' x, to exercise the replay-not-identical path
    const slot = this.abi.feed_slots.x;
    for (const id of ids) this.device.queue.writeBuffer(this.lanes[this.laneOf(slot)], this.byteAt(slot) + id * 4, new Float32Array([dx]));
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); }
  destroy() { for (const b of [...(this.lanes || []), this.head, this.pts]) b?.destroy(); for (const s of this.staging || []) s.buffer.destroy(); this.device.destroy(); }
}

// ---------------------------------------------------------------- wasm route
class WasmProduct {
  kind = "wasm";

  async init(rec, manifest) {
    this.rec = rec; this.n = rec.n; this.pad = rec.pad; this.manifest = manifest;
    this.program = await TuringGraphProgram.load(rec.wasm, rec.wasm_abi);
    const abi = this.program.abi;
    this.inputs = {}; this.nexts = {};
    for (const name of abi.argument_names) this.inputs[name] = this.program.view(abi.inputs[name]);
    for (const out of abi.outputs) this.nexts[out.name] = this.program.view(out.id);
    this.state = manifest.law.state;
    this.ndisp = this.pad;
    this.packed = new ArrayBuffer(this.pad * 16);
    this.f32 = new Float32Array(this.packed); this.u32 = new Uint32Array(this.packed);
  }

  reset({ named, uniforms }) {
    for (const view of Object.values(this.inputs)) if (view.length > 1) view.fill(0);
    for (let i = 0; i < this.pad; i++) this.inputs.pid[i] = i;
    for (const [name, values] of Object.entries(named)) if (this.inputs[name] && this.inputs[name].length >= values.length) this.inputs[name].set(values);
    for (const [name, value] of Object.entries(uniforms)) this.inputs[name][0] = value;
  }

  setUniform(name, value) { this.inputs[name][0] = value; }
  setColours(named) { for (const [name, values] of Object.entries(named)) this.inputs[name].set(values); }

  step(count) {
    for (let k = 0; k < count; k++) {
      this.program.run();
      for (const name of this.state) this.inputs[name].set(this.nexts[name + "_next"]);
    }
  }

  points() {
    const p = this.manifest.points, I = this.inputs, O = this.nexts, f = this.f32, u = this.u32, rate = this.rec.emit_rate;
    const x = I[p.x], y = I[p.y], vx = I[p.vx], vy = I[p.vy], age = I[p.live[0]];
    const r = O[p.rgb[0]], g = O[p.rgb[1]], b = O[p.rgb[2]], rad = O[p.radius], rest2 = p.rest_speed * p.rest_speed;
    const byte = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
    for (let i = 0; i < this.pad; i++) {
      f[4 * i] = x[i]; f[4 * i + 1] = y[i];
      u[4 * i + 2] = (255 << 24) | (byte(b[i]) << 16) | (byte(g[i]) << 8) | byte(r[i]);
      f[4 * i + 3] = age[i] >= i * rate ? ((vx[i] * vx[i] + vy[i] * vy[i]) < rest2 ? 200 : 100) + rad[i] * 1000 : 0;
    }
    return f;
  }

  async columnsNow(names) {
    const n = this.n;
    return Object.fromEntries(names.map((name) => [name, Float32Array.from((this.inputs[name] || this.nexts[name]).subarray(0, n))]));
  }
  nudge(ids, dx) { for (const id of ids) this.inputs.x[id] = dx; }
  async idle() {}
  destroy() {}
}

// ------------------------------------------------------------------ runtime
export class ComputeRuntime {
  constructor({ onStatus, onSamples, onSupport }) {
    this.onStatus = onStatus; this.onSamples = onSamples; this.onSupport = onSupport || (() => {});
    this.spec = null; this.manifest = null; this.product = null; this.productKey = null;
    this.epoch = null; this.uniforms = {}; this.support = null;
    this.named = {}; this.raster = null; this.sampleSpec = null; this.rest = null; this.verdict = null;
    this.restPending = false; this.stepMs = 0; this.stepsPerFrame = 0; this.recovering = false;
    this.steps = 0; this.running = false; this.last = 0; this.fps = 0; this.frames = 0; this.fpsAt = 0;
    this.route = ""; this.why = ""; this.warn = false; this.queue = Promise.resolve(); this.error = null;
    this.ready = false; this.note = "";
  }

  get active() { return !!this.spec; }
  get rec() { return this.manifest && this.spec ? this.manifest.sizes[this.spec.size] : null; }

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
    if (!this.manifest) { this.manifest = await json(spec.manifest); await this.#capabilities(); }
    const key = `${spec.manifest}:${spec.size}`;
    const verdict = this.support[spec.size];
    if (key !== this.productKey) {
      if (verdict && !verdict.ok) throw new Error(`${spec.size} refused: ${verdict.reason}`);
      await this.#load(spec, key);
      this.epoch = null;
    }
    if (spec.epoch !== this.epoch) {
      this.epoch = spec.epoch; this.uniforms = { ...spec.uniforms };
      this.#reset();
    } else {
      for (const [name, value] of Object.entries(spec.uniforms)) {
        if (this.uniforms[name] !== value) { this.uniforms[name] = value; this.product.setUniform(name, value); }
      }
    }
  }

  async #load(spec, key) {
    this.ready = false; this.running = false; this.error = null;
    if (this.product) { this.product.destroy(); this.product = null; }
    const rec = this.manifest.sizes[spec.size];
    // The compiler decides (rec.route, from its verdict); the browser can only veto.
    const v = rec.verdict, model = this.manifest.cost_model;
    const detail = `n=${rec.n.toLocaleString()}, ${rec.wgsl_abi.dispatches} dispatches, ${mb(rec.memory.arena_bytes)}: predicted ${v.speedup.toFixed(2)}x, needs ${model.min_speedup.toFixed(1)}x`;
    let product = null; this.warn = false;
    if (rec.route === "webgpu-compute" && navigator.gpu) {
      try { product = new GpuProduct(); await product.init(rec, this.manifest); this.why = `compiler verdict: the dispatches pay (${detail})`; }
      catch (error) {
        try { product?.destroy(); } catch { /* the device may be gone */ }
        product = null; this.warn = true;
        if (!rec.wasm) throw error;
        this.why = `FALLBACK to wasm: WebGPU could not run the generated program (${error.message || error})`;
      }
    } else if (rec.route === "webgpu-compute") {
      if (!rec.wasm) throw new Error("needs WebGPU: this size has no wasm product");
      this.warn = true; this.why = `FALLBACK to wasm: the compiler chose WebGPU compute (${detail}) but navigator.gpu is absent here (needs https and a WebGPU browser)`;
    } else this.why = `compiler verdict: the dispatches do not pay (${detail})`;
    if (!product) { product = new WasmProduct(); await product.init(rec, this.manifest); }
    this.product = product; this.productKey = key;
    this.route = product.kind;
    this.ready = true;
  }

  #reset(phase2Rest) {
    this.product.reset({ named: this.named, uniforms: { ...this.rec.uniforms, ...this.uniforms } });
    this.steps = 0; this.running = true; this.last = performance.now() / 1000;
    this.restPending = true;
    if (!phase2Rest) this.note = "";
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
    if (this.rest) this.#sample(this.rest); else { this.note = "image loaded: it is mapped when the pile has settled"; this.onStatus(); }
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

  // --- per frame ---
  frame(now) {
    if (!this.ready || !this.running || !this.spec) return;
    const elapsed = Math.min(0.1, Math.max(0, now - this.last));
    this.last = now;
    this.frames++;
    if (now - this.fpsAt > 0.5) { this.fps = this.frames / (now - this.fpsAt); this.frames = 0; this.fpsAt = now; this.onStatus(); }
    const dt = this.manifest.law.dt;
    // Both phases stop at EXACTLY the same step, so their rest positions are comparable
    const total = Math.ceil(this.manifest.settle_seconds / dt - 1e-9);
    const settled = this.steps >= total;
    if (settled && this.restPending) { this.restPending = false; this.#recordRest(); }
    // the sim runs in real time at speed 1; a slow route takes what its frame budget allows
    const wanted = settled || this.product.busy ? 0
      : Math.min(total - this.steps, Math.max(1, Math.min(48, Math.round(elapsed / dt * (this.spec.speed || 1)))));
    const budget = this.product.kind === "wasm" ? 22 : 1e9;
    const t0 = performance.now();
    let done = 0;
    while (done < wanted && (performance.now() - t0) < budget) { const chunk = this.product.kind === "wasm" ? 1 : wanted; this.product.step(chunk); done += chunk; }
    if (done) { this.steps += done; this.stepsPerFrame = done; if (this.product.kind === "wasm") this.stepMs = (performance.now() - t0) / done; }
    else if (!this.packedAtRest) { this.product.step(0); this.packedAtRest = true; }
    if (done) this.packedAtRest = false;
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
    const dt = this.manifest.law.dt, t = this.steps * dt, span = this.manifest.settle_seconds;
    const done = t >= span ? "settled" : `${Math.round(100 * t / span)}%`;
    const fail = this.product.failure ? ` | GPU error: ${this.product.failure}` : "";
    const ms = this.product.kind === "wasm" ? this.stepMs : this.product.stepMs;
    const rate = ms ? ` | ${ms.toFixed(2)} ms/step` : "";
    return `${this.route}${adapter}: ${this.why} | t ${t.toFixed(1)} s ${done} | ${this.fps.toFixed(0)} fps${rate}${this.note ? " | " + this.note : ""}${this.verdictText()}${fail}`;
  }

  // test/diagnostics only
  advance(count) {
    const total = Math.ceil(this.manifest.settle_seconds / this.manifest.law.dt - 1e-9);
    count = Math.min(count, total - this.steps);
    this.product.step(count); this.steps += count; this.packedAtRest = false; return this.steps;
  }
  columnsNow(names) { return this.product.columnsNow(names); }
}

// --------------------------------------------------- GL points (shared scene)
const POINT_VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform mat4 uViewProj; uniform vec4 uMap; uniform vec3 uBoard; uniform float uPx; uniform float uDpr; uniform float uMax; uniform float uStride;
out vec4 vColor; out float vBig;
void main() {
  if (aMeta < 50.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vColor = vec4(0.0); vBig = 0.0; return; }
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  // world -> board with ONE scale (uMap: centre x, centre y, scale k): circles stay circles at every size
  gl_Position = uViewProj * vec4(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z, 1.0);
  float diameter = 2.0 * radius * uMap.z * uPx * sqrt(uStride);
  float least = flag > 1.5 ? 1.5 * uDpr : 3.5 * uDpr;      // a flying particle is never a speck
  gl_PointSize = min(max(diameter, least), uMax);
  vBig = step(5.0, gl_PointSize);
  vColor = aColor;
}`;
const POINT_FS = `#version 300 es
precision highp float;
in vec4 vColor; in float vBig; out vec4 FragColor;
uniform vec3 u_background;   // the page's shared background: what an unfilled (alpha < 1) point fades to
void main() {
  if (vBig > 0.5 && length(gl_PointCoord - 0.5) > 0.5) discard;
  FragColor = vec4(mix(u_background, vColor.rgb, vColor.a), 1.0);
}`;

export class PointCloud {
  constructor(gl, link, compile) {
    this.gl = gl;
    this.program = link(compile(gl.VERTEX_SHADER, POINT_VS), compile(gl.FRAGMENT_SHADER, POINT_FS));
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.dyn = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 16, 8);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 16, 12);
    gl.bindVertexArray(null);
    this.max = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.loc = Object.fromEntries(["uViewProj", "uMap", "uBoard", "uPx", "uDpr", "uMax", "uStride", "u_background"].map((n) => [n, gl.getUniformLocation(this.program, n)]));
  }

  // data: packed points ([x, y, rgba8, meta] per displayed id); view: the size's world window
  draw(data, { viewProj, view, board, y, pxPerUnit, stride, background }) {
    const gl = this.gl;
    if (!data) return;
    const [x0, x1, y0, y1] = view, k = Math.min(board[0] / (x1 - x0), board[1] / (y1 - y0));
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.uViewProj, false, viewProj);
    gl.uniform4f(this.loc.uMap, (x0 + x1) / 2, (y0 + y1) / 2, k, 0);
    gl.uniform3f(this.loc.uBoard, board[0], board[1], y);
    gl.uniform1f(this.loc.uPx, pxPerUnit); gl.uniform1f(this.loc.uDpr, window.devicePixelRatio || 1); gl.uniform1f(this.loc.uMax, this.max);
    gl.uniform1f(this.loc.uStride, stride || 1);
    gl.uniform3fv(this.loc.u_background, new Float32Array(background || [0.085, 0.085, 0.1]));
    gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.drawArrays(gl.POINTS, 0, data.length / 4);
    gl.depthMask(true); gl.bindVertexArray(null);
  }
}
