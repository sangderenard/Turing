// The page's generic COMPUTE-PRODUCT runtime.
//
// A game declares a compute product in its state:
//
//   game.compute = { manifest, size, epoch, layout, uniforms, speed, named }
//
// `manifest` is a compiler-built products manifest (fountain_build.py writes
// one): per size, the generated WGSL program + its ABI (`wgsl`, `wgsl_abi`:
// dispatch entry points over ONE arena buffer, the arena slot of every formal
// and output), the wasm product (`wasm`, `wasm_abi`), the compiler's
// deployment verdict (`route`, `verdict`), and how to present it (`points`,
// `layout_columns`).  This runtime creates the device, keeps the arena
// resident, runs the program's dispatches each step, and falls back to the
// wasm product when the classifier said the dispatches do not pay or
// `navigator.gpu` is missing.  Nothing here knows what the law is.
//
// Presentation: the main renderer is WebGL2, which cannot read WebGPU
// buffers.  The product's `points` description (position, velocity for the
// at-rest flag, colour, liveness) is packed on the device by one small
// generic compute pass into 16 bytes per id -- [x, y, rgba8 (as bits), meta]
// -- and that packed buffer alone is copied back (triple buffered, one or two
// frames late) for WebGL to draw as GL points in the shared scene.  No WGSL
// is read by a vertex stage, so devices with no vertex-stage storage buffers
// (Apple's) are not a case.  The wasm route packs the same layout on the CPU.
//
// Determinism: a product's reset writes the same bytes every time, and the
// law has no atomics and no scatter, so a replay from the same initial
// columns on the same device repeats exactly.  The runtime records the rest
// positions when phase 1 settles (one readback, an event) and compares them
// with phase 2's.
import { TuringGraphProgram } from "./turing-wasm-shell.js";

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

// ---------------------------------------------------------------- GPU route
const PACK_WGSL = `
@group(0) @binding(0) var<storage, read> arena: array<f32>;
@group(0) @binding(1) var<storage, read_write> pts: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= __N__u) { return; }
  let live = arena[__AGE__u * __N__u + i] >= arena[__START__u * __N__u + i];
  let vx = arena[__VX__u * __N__u + i];
  let vy = arena[__VY__u * __N__u + i];
  let rest = (vx * vx + vy * vy) < __REST2__;
  let c = pack4x8unorm(vec4<f32>(arena[__R__u * __N__u + i], arena[__G__u * __N__u + i], arena[__B__u * __N__u + i], 1.0));
  pts[4u * i] = arena[__X__u * __N__u + i];
  pts[4u * i + 1u] = arena[__Y__u * __N__u + i];
  pts[4u * i + 2u] = bitcast<f32>(c);
  pts[4u * i + 3u] = select(0.0, select(1.0, 2.0, rest), live);
}`;

class GpuProduct {
  kind = "webgpu-compute";

  async init(rec, manifest) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    this.device = await adapter.requestDevice();
    this.device.addEventListener?.("uncapturederror", (e) => { this.failure = String((e.error && e.error.message) || e.error); });
    this.adapterInfo = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.device].filter(Boolean).join("/") : "";
    this.rec = rec; this.n = rec.n; this.abi = rec.wgsl_abi; this.manifest = manifest;
    this.state = manifest.law.state;
    const module = this.device.createShaderModule({ code: await fetch(rec.wgsl).then((r) => r.text()) });
    this.#check(await module.getCompilationInfo(), "generated WGSL");
    const storage = (type) => ({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = this.device.createBindGroupLayout({ entries: [storage("storage")] });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipelines = await Promise.all(this.abi.entries.map((entryPoint) =>
      this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } })));
    const slot = (name) => this.abi.feed_slots[name], oslot = (name) => this.abi.output_slots[name], p = manifest.points;
    const code = PACK_WGSL.replaceAll("__N__", String(this.n))
      .replaceAll("__AGE__", slot(p.live[0])).replaceAll("__START__", slot(p.live[1]))
      .replaceAll("__X__", slot(p.x)).replaceAll("__Y__", slot(p.y)).replaceAll("__VX__", slot(p.vx)).replaceAll("__VY__", slot(p.vy))
      .replaceAll("__R__", oslot(p.rgb[0])).replaceAll("__G__", oslot(p.rgb[1])).replaceAll("__B__", oslot(p.rgb[2]))
      .replaceAll("__REST2__", (p.rest_speed * p.rest_speed).toFixed(8));
    const pack = this.device.createShaderModule({ code });
    this.#check(await pack.getCompilationInfo(), "pack WGSL");
    this.pack = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module: pack, entryPoint: "main" } });
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.arena = this.device.createBuffer({ size: this.abi.slots * this.n * 4, usage: S });
    this.pts = this.device.createBuffer({ size: this.n * 16, usage: S });
    this.group = this.device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: this.arena } }] });
    this.packGroup = this.device.createBindGroup({ layout: this.pack.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.arena } }, { binding: 1, resource: { buffer: this.pts } }] });
    this.staging = [0, 1, 2].map(() => ({ buffer: this.device.createBuffer({ size: this.n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
    // each state output is committed over its feed slot after the step: one more dispatch, the
    // arena's disjoint slots (WebGPU forbids a buffer-to-buffer copy within one buffer)
    const commit = this.state.map((name) => `  arena[${slot(name) * this.n}u + i] = arena[${oslot(name + "_next") * this.n}u + i];`).join("\n");
    const commitModule = this.device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= ${this.n}u) { return; }
${commit}
}` });
    this.#check(await commitModule.getCompilationInfo(), "commit WGSL");
    this.pipelines.push(await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: commitModule, entryPoint: "commit" } }));
    this.host = new Float32Array(this.abi.slots * this.n);
    this.latest = null;
  }

  #check(info, what) {
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(`${what} rejected: ` + errors.map((m) => `${m.lineNum}: ${m.message}`).join(" | "));
  }

  reset({ columns, init, named, uniforms }) {
    const n = this.n, host = this.host, S = this.abi.feed_slots;
    host.fill(0);
    for (const [name, value] of Object.entries(init)) host.fill(value, S[name] * n, S[name] * n + n);
    for (const name of this.abi.arguments) {
      const source = columns[name] || named[name];
      if (source && !this.abi.uniforms.includes(name)) host.set(source, S[name] * n);
    }
    for (const [name, value] of Object.entries(uniforms)) host[S[name] * n] = value;
    this.device.queue.writeBuffer(this.arena, 0, host);
    this.latest = null;
  }

  setUniform(name, value) { this.device.queue.writeBuffer(this.arena, this.abi.feed_slots[name] * this.n * 4, new Float32Array([value])); }

  step(count) {
    const encoder = this.device.createCommandEncoder(), groups = this.abi.groups[0];
    for (let k = 0; k < count; k++) {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, this.group);
      for (const pipeline of this.pipelines) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(groups); }
      pass.end();
    }
    const pack = encoder.beginComputePass();
    pack.setPipeline(this.pack); pack.setBindGroup(0, this.packGroup);
    pack.dispatchWorkgroups(Math.ceil(this.n / 64));
    pack.end();
    // back-pressure and a measured cost: the frame loop does not submit more while the last
    // submission is unfinished, and the time to finish it, per step, is the step cost
    const submitted = performance.now();
    this.pending = (this.pending || 0) + 1;
    const settle = () => {
      this.pending--;
      if (count > 0) { const per = (performance.now() - submitted) / count; this.stepMs = this.stepMs ? this.stepMs * 0.8 + per * 0.2 : per; }
    };
    const free = this.staging.find((s) => !s.busy);
    if (free) {
      free.busy = true;
      encoder.copyBufferToBuffer(this.pts, 0, free.buffer, 0, this.n * 16);
      this.device.queue.submit([encoder.finish()]);
      free.buffer.mapAsync(GPUMapMode.READ).then(() => {
        this.latest = new Float32Array(free.buffer.getMappedRange().slice(0));
        free.buffer.unmap(); free.busy = false;
      }).catch(() => { free.busy = false; });
    } else this.device.queue.submit([encoder.finish()]);
    this.device.queue.onSubmittedWorkDone().then(settle);
  }

  get busy() { return (this.pending || 0) > 0; }

  points() { return this.latest; }

  async feedsNow() {          // an event, never per frame: the whole arena, by name
    const bytes = this.abi.slots * this.n * 4;
    const read = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.arena, 0, read, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
    const at = (slot) => all.slice(slot * this.n, (slot + 1) * this.n);
    return { ...Object.fromEntries(this.abi.arguments.map((name) => [name, at(this.abi.feed_slots[name])])),
             ...Object.fromEntries(this.abi.outputs.map((name) => [name, at(this.abi.output_slots[name])])) };
  }

  async idle() { await this.device.queue.onSubmittedWorkDone(); }
  destroy() { for (const b of [this.arena, this.pts]) b.destroy(); for (const s of this.staging) s.buffer.destroy(); this.device.destroy(); }
}

// ---------------------------------------------------------------- wasm route
class WasmProduct {
  kind = "wasm";

  async init(rec, manifest) {
    this.rec = rec; this.n = rec.n; this.manifest = manifest;
    this.program = await TuringGraphProgram.load(rec.wasm, rec.wasm_abi);
    const abi = this.program.abi;
    this.inputs = {}; this.nexts = {};
    for (const name of abi.argument_names) this.inputs[name] = this.program.view(abi.inputs[name]);
    for (const out of abi.outputs) this.nexts[out.name] = this.program.view(out.id);
    this.state = manifest.law.state;
    this.packed = new ArrayBuffer(this.n * 16);
    this.f32 = new Float32Array(this.packed); this.u32 = new Uint32Array(this.packed);
  }

  reset({ columns, init, named, uniforms }) {
    for (const view of Object.values(this.inputs)) if (view.length > 1) view.fill(0);
    for (const [name, value] of Object.entries(init)) this.inputs[name].fill(value);
    for (const name of Object.keys(this.inputs)) {
      const source = columns[name] || named[name];
      if (source && this.inputs[name].length === source.length) this.inputs[name].set(source);
    }
    for (const [name, value] of Object.entries(uniforms)) this.inputs[name][0] = value;
  }

  setUniform(name, value) { this.inputs[name][0] = value; }

  step(count) {
    for (let k = 0; k < count; k++) {
      this.program.run();
      for (const name of this.state) this.inputs[name].set(this.nexts[name + "_next"]);
    }
  }

  points() {
    const p = this.manifest.points, I = this.inputs, O = this.nexts, f = this.f32, u = this.u32;
    const x = I[p.x], y = I[p.y], vx = I[p.vx], vy = I[p.vy], age = I[p.live[0]], start = I[p.live[1]];
    const r = O[p.rgb[0]], g = O[p.rgb[1]], b = O[p.rgb[2]], rest2 = p.rest_speed * p.rest_speed;
    const byte = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
    for (let i = 0; i < this.n; i++) {
      f[4 * i] = x[i]; f[4 * i + 1] = y[i];
      u[4 * i + 2] = (255 << 24) | (byte(b[i]) << 16) | (byte(g[i]) << 8) | byte(r[i]);
      f[4 * i + 3] = age[i] >= start[i] ? ((vx[i] * vx[i] + vy[i] * vy[i]) < rest2 ? 2 : 1) : 0;
    }
    return f;
  }

  async feedsNow() {
    return { ...Object.fromEntries(Object.entries(this.inputs).map(([k, v]) => [k, Float32Array.from(v)])),
             ...Object.fromEntries(Object.entries(this.nexts).map(([k, v]) => [k, Float32Array.from(v)])) };
  }
  async idle() {}
  destroy() {}
}

// ------------------------------------------------------------------ runtime
export class ComputeRuntime {
  constructor({ onStatus, onSamples }) {
    this.onStatus = onStatus; this.onSamples = onSamples;
    this.spec = null; this.manifest = null; this.product = null; this.productKey = null;
    this.layout = null; this.layoutUrl = null; this.epoch = null; this.uniforms = {};
    this.named = {}; this.raster = null; this.sampleSpec = null; this.rest = null; this.determinism = null;
    this.restPending = false; this.stepMs = 0; this.stepsPerFrame = 0;
    this.steps = 0; this.running = false; this.last = 0; this.fps = 0; this.frames = 0; this.fpsAt = 0;
    this.route = ""; this.why = ""; this.warn = false; this.queue = Promise.resolve(); this.error = null;
    this.sizes = null; this.ready = false; this.note = "";
  }

  get active() { return !!this.spec; }

  // Called with each game state: the declared product (or null).
  sync(spec) {
    if (!spec) { this.spec = null; this.running = false; return; }
    this.spec = spec;
    this.queue = this.queue.then(() => this.#apply(spec)).catch((error) => { this.error = String(error.message || error); console.error("[compute]", error); });
  }

  async #apply(spec) {
    const key = `${spec.manifest}:${spec.size}`;
    if (key !== this.productKey) await this.#load(spec, key);
    let relayout = false;
    if (spec.layout !== this.layoutUrl) {
      this.layout = await this.#fetchLayout(spec.layout);
      this.layoutUrl = spec.layout; relayout = true;
      this.sizes = this.layout.radius;
      this.rest = null;                          // a new pile: its rest positions are not known yet
    }
    if (relayout || spec.epoch !== this.epoch) {
      this.epoch = spec.epoch; this.uniforms = { ...spec.uniforms };
      this.#reset();
    } else {
      for (const [name, value] of Object.entries(spec.uniforms)) {
        if (this.uniforms[name] !== value) { this.uniforms[name] = value; this.product.setUniform(name, value); }
      }
    }
  }

  async #load(spec, key) {
    this.ready = false; this.running = false;
    if (this.product) { this.product.destroy(); this.product = null; }
    this.manifest = await json(spec.manifest);
    const rec = this.manifest.sizes[spec.size];
    // The compiler decides (rec.route, from its verdict); the browser can only veto.
    const v = rec.verdict, model = this.manifest.cost_model;
    const detail = `n=${rec.n}, ${rec.wgsl_abi.dispatches} dispatches: predicted ${v.speedup.toFixed(2)}x, needs ${model.min_speedup.toFixed(1)}x`;
    let product = null; this.warn = false;
    if (rec.route === "webgpu-compute") {
      if (navigator.gpu) {
        try { product = new GpuProduct(); await product.init(rec, this.manifest); this.why = `compiler verdict: the dispatches pay (${detail})`; }
        catch (error) { product = null; this.warn = true; this.why = `FALLBACK to wasm: WebGPU could not run the generated program (${error.message || error})`; }
      } else { this.warn = true; this.why = `FALLBACK to wasm: the compiler chose WebGPU compute (${detail}) but navigator.gpu is absent here (needs https and a WebGPU browser)`; }
    } else this.why = `compiler verdict: the dispatches do not pay (${detail})`;
    if (!product) { product = new WasmProduct(); await product.init(rec, this.manifest); }
    this.product = product; this.rec = rec; this.productKey = key;
    this.layoutUrl = null; this.epoch = null;
    this.route = product.kind;
    this.ready = true;
  }

  async #fetchLayout(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`layout ${response.status}`);
    const all = new Float32Array(await response.arrayBuffer());
    const names = this.manifest.layout_columns, n = this.rec.n;
    if (all.length !== names.length * n) throw new Error(`layout has ${all.length} values, expected ${names.length * n}`);
    return Object.fromEntries(names.map((name, k) => [name, all.subarray(k * n, (k + 1) * n)]));
  }

  #reset() {
    this.product.reset({ columns: this.layout, init: this.manifest.initial, named: this.named, uniforms: { ...this.rec.uniforms, ...this.uniforms } });
    this.steps = 0; this.running = true; this.last = performance.now() / 1000;
    this.restPending = true; this.note = "";
  }

  // --- image -> per-id feeds (a game's `file` control with a `sample` spec) ---
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
    this.raster = raster; this.sampleSpec = spec;
    if (this.rest) this.#sample(); else { this.note = "image loaded: it is mapped when the pile has settled"; this.onStatus(); }
  }

  // The image is ONLY sampled into the per-id feeds, at each id's ACTUAL rest position
  // in phase 1's pile (normalised by the pile's own extent); nothing of it is drawn.
  // (fountain.fit_extent + fountain.sample_raster are the same formulas.)
  #sample() {
    const { w, h, data } = this.raster, spec = this.sampleSpec, X = this.rest.x, Y = this.rest.y, n = X.length;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { x0 = Math.min(x0, X[i]); x1 = Math.max(x1, X[i]); y0 = Math.min(y0, Y[i]); y1 = Math.max(y1, Y[i]); }
    const bytes = new Uint8Array(3 * n);
    this.named = Object.fromEntries(spec.into.map((name) => [name, new Float32Array(n)]));
    for (let i = 0; i < n; i++) {
      const u = Math.fround((X[i] - x0) / Math.max(x1 - x0, 1e-12)), v = Math.fround(1 - (Y[i] - y0) / Math.max(y1 - y0, 1e-12));
      const px = Math.min(Math.max(Math.floor(u * w), 0), w - 1), py = Math.min(Math.max(Math.floor(v * h), 0), h - 1);
      const at = 4 * (py * w + px);
      spec.into.forEach((name, c) => { bytes[3 * i + c] = data[at + c]; this.named[name][i] = Math.fround(data[at + c] / 255); });
    }
    let text = "";
    for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    this.note = "image mapped at the rest positions";
    this.onSamples(spec.post, btoa(text));
  }

  // When the pile has settled (an event): record phase 1's rest positions; in phase 2
  // compare them with phase 1's.
  async #recordRest() {
    const feeds = await this.product.feedsNow();
    const x = Float32Array.from(feeds.x), y = Float32Array.from(feeds.y);
    if (!this.uniforms.use_image) {
      this.rest = { x, y };
      this.determinism = null;
      if (this.raster) this.#sample();
    } else if (this.rest) {
      let equal = 0, worst = 0;
      for (let i = 0; i < x.length; i++) {
        if (x[i] === this.rest.x[i] && y[i] === this.rest.y[i]) equal++;
        worst = Math.max(worst, Math.abs(x[i] - this.rest.x[i]), Math.abs(y[i] - this.rest.y[i]));
      }
      this.determinism = { equal, n: x.length, worst };
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
    // the sim runs in real time at speed 1; a slow route (wasm) takes what its frame budget allows
    const wanted = settled || this.product.busy ? 0
      : Math.min(total - this.steps, Math.max(1, Math.min(48, Math.round(elapsed / dt * (this.spec.speed || 1)))));
    const budget = this.product.kind === "wasm" ? 22 : 1e9;
    const t0 = performance.now();
    let done = 0;
    while (done < wanted && (performance.now() - t0) < budget) { const chunk = Math.min(wanted - done, this.product.kind === "wasm" ? 1 : wanted); this.product.step(chunk); done += chunk; }
    if (done) { this.steps += done; this.stepsPerFrame = done; if (this.product.kind === "wasm") this.stepMs = (performance.now() - t0) / done; }
    else if (!this.packedAtRest) { this.product.step(0); this.packedAtRest = true; }
    if (done) this.packedAtRest = false;
  }

  statusText() {
    if (!this.ready) return this.error ? "compute: " + this.error : "compute: loading";
    const adapter = this.product.adapterInfo ? ` [${this.product.adapterInfo}]` : "";
    const dt = this.manifest.law.dt, t = this.steps * dt, span = this.manifest.settle_seconds;
    const done = t >= span ? "settled" : `${Math.round(100 * t / span)}%`;
    const fail = this.product.failure ? ` | GPU error: ${this.product.failure}` : "";
    const det = this.determinism ? ` | replay rest positions identical: ${this.determinism.equal}/${this.determinism.n}${this.determinism.equal === this.determinism.n ? "" : " (max diff " + this.determinism.worst.toExponential(2) + ")"}` : "";
    const ms = this.product.kind === "wasm" ? this.stepMs : this.product.stepMs;
    const rate = ms ? ` | ${ms.toFixed(2)} ms/step` : "";
    return `${this.route}${adapter}: ${this.why} | t ${t.toFixed(1)} s ${done} | ${this.fps.toFixed(0)} fps${rate}${this.note ? " | " + this.note : ""}${det}${fail}`;
  }

  // test/diagnostics only
  advance(count) {
    const total = Math.ceil(this.manifest.settle_seconds / this.manifest.law.dt - 1e-9);
    count = Math.min(count, total - this.steps);
    this.product.step(count); this.steps += count; this.packedAtRest = false; return this.steps;
  }
  feedsNow() { return this.product.feedsNow(); }
}

// --------------------------------------------------- GL points (shared scene)
const POINT_VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
layout(location=3) in float aRadius;
uniform mat4 uViewProj; uniform vec4 uWorld; uniform vec3 uBoard; uniform float uPx; uniform float uDpr; uniform float uMax;
out vec4 vColor; out float vBig;
void main() {
  if (aMeta < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vColor = vec4(0.0); vBig = 0.0; return; }
  float u = (aPos.x - uWorld.x) / (uWorld.y - uWorld.x);
  float v = 1.0 - (aPos.y - uWorld.z) / (uWorld.w - uWorld.z);
  gl_Position = uViewProj * vec4(u * uBoard.x, uBoard.z, v * uBoard.y, 1.0);
  float diameter = 2.0 * aRadius * (uBoard.x / (uWorld.y - uWorld.x)) * uPx;
  float least = aMeta > 1.5 ? 1.5 * uDpr : 3.5 * uDpr;      // a flying particle is never a speck
  gl_PointSize = min(max(diameter, least), uMax);
  vBig = step(5.0, gl_PointSize);
  vColor = aColor;
}`;
const POINT_FS = `#version 300 es
precision highp float;
in vec4 vColor; in float vBig; out vec4 FragColor;
void main() {
  if (vBig > 0.5 && length(gl_PointCoord - 0.5) > 0.5) discard;
  FragColor = vec4(vColor.rgb, 1.0);
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
    this.size = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.size);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.sizeFor = null; this.count = 0;
    this.max = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.loc = Object.fromEntries(["uViewProj", "uWorld", "uBoard", "uPx", "uDpr", "uMax"].map((n) => [n, gl.getUniformLocation(this.program, n)]));
  }

  // data: packed points ([x, y, rgba8, meta] per id); radius: per-id world radius
  draw(data, radius, { viewProj, world, board, y, pxPerUnit }) {
    const gl = this.gl;
    if (!data || !radius) return;
    gl.bindVertexArray(this.vao);
    if (this.sizeFor !== radius) { gl.bindBuffer(gl.ARRAY_BUFFER, this.size); gl.bufferData(gl.ARRAY_BUFFER, radius, gl.STATIC_DRAW); this.sizeFor = radius; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.uViewProj, false, viewProj);
    gl.uniform4fv(this.loc.uWorld, world);
    gl.uniform3f(this.loc.uBoard, board[0], board[1], y);
    gl.uniform1f(this.loc.uPx, pxPerUnit); gl.uniform1f(this.loc.uDpr, window.devicePixelRatio || 1); gl.uniform1f(this.loc.uMax, this.max);
    gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.drawArrays(gl.POINTS, 0, radius.length);
    gl.depthMask(true); gl.bindVertexArray(null);
  }
}
