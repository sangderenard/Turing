// Fountain page.  The particle law is not written here: it is the compiler's product
// of examples/meowdoku/fountain.py (fountain_step), in two forms --
//   fountain_<tag>.wgsl   a WebGPU compute shader generated from the traced law
//                         (ssa_webgpu_backend.emit_tensor_program_module), and
//   fountain_<tag>.wasm   the same law as buffer-table WebAssembly.
// The compiler's deployment verdict (fountain.manifest.json, from
// webgpu_compute_payoff) says which one this particle count should run on;
// the browser can only veto it (no navigator.gpu).  This file owns the host
// boundary: resident buffers, the dispatch loop, picking the image, drawing.
import { TuringGraphProgram } from "./turing-wasm-shell.js";

const $ = (id) => document.getElementById(id);
const DT = 1 / 120;
const SPECIES_NAMES = ["grain", "bead", "dust", "pearl"];

const app = {
  manifest: null, tag: "full", rec: null, seed: 7, mix: 0.5, mode: 0, speed: 2,
  layout: null, raster: null, colours: null, engine: null, route: null, steps: 0,
  phase: "idle", useImage: 0, running: false, last: 0, frames: 0, fps: 0, fpsAt: 0, failure: null,
};

// ------------------------------------------------------------------ server
async function api(path, body) {
  try {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return await response.json();
  } catch (error) { return { error: String(error) }; }
}
const act = (body) => api("/api/action", body);

async function fetchLayout() {
  const rec = app.rec;
  const url = `/api/fountain/layout?cols=${rec.cols}&rows=${rec.rows}&seed=${app.seed}&mix=${app.mix}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`layout ${response.status}`);
  const all = new Float32Array(await response.arrayBuffer());
  const names = app.manifest.layout_columns, n = rec.n;
  if (all.length !== names.length * n) throw new Error(`layout has ${all.length} values, expected ${names.length * n}`);
  const layout = {};
  names.forEach((name, k) => { layout[name] = all.subarray(k * n, (k + 1) * n); });
  return layout;
}

// ------------------------------------------------------------------- image
// The raster is the image fitted to the basin (cover), RASTER_W x RASTER_H bytes;
// an id's colour is the raster pixel at its slot (fountain.sample_raster is the same formula).
function rasterSize() { return [2 * app.rec.cols, 2 * app.rec.rows]; }

async function rasterize(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const [w, h] = rasterSize();
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const scale = Math.max(w / bitmap.width, h / bitmap.height);
  const dw = bitmap.width * scale, dh = bitmap.height * scale;
  ctx.drawImage(bitmap, (w - dw) / 2, (h - dh) / 2, dw, dh);
  const thumb = $("thumb");
  thumb.width = 84; thumb.height = Math.round(84 * h / w);
  thumb.getContext("2d").drawImage(canvas, 0, 0, thumb.width, thumb.height);
  thumb.style.display = "block";
  if (bitmap.close) bitmap.close();
  return { w, h, data: ctx.getImageData(0, 0, w, h).data };
}

function sampleRaster(raster, u, v) {
  const { w, h, data } = raster;
  const n = u.length, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const px = Math.min(Math.max(Math.floor(u[i] * w), 0), w - 1);
    const py = Math.min(Math.max(Math.floor(v[i] * h), 0), h - 1);
    const at = 4 * (py * w + px);
    out[i] = Math.fround(data[at] / 255);
    out[n + i] = Math.fround(data[at + 1] / 255);
    out[2 * n + i] = Math.fround(data[at + 2] / 255);
  }
  return out;
}

function mapImage() {
  // the per-id colour buffer: the raster sampled at every id's recorded slot
  const n = app.rec.n;
  const all = sampleRaster(app.raster, app.layout.slot_u, app.layout.slot_v);
  app.colours = { ir: all.subarray(0, n), ig: all.subarray(n, 2 * n), ib: all.subarray(2 * n, 3 * n) };
  const bytes = new Uint8Array(3 * n);
  for (let i = 0; i < n; i++) {
    bytes[3 * i] = Math.round(app.colours.ir[i] * 255);
    bytes[3 * i + 1] = Math.round(app.colours.ig[i] * 255);
    bytes[3 * i + 2] = Math.round(app.colours.ib[i] * 255);
  }
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  act({ type: "fountain_image", rgb: btoa(text) });
  $("replay").disabled = false;
}

// ----------------------------------------------------------------- engines
const worldView = () => app.manifest.view;   // x0 x1 y0 y1

function fit(width, height) {
  const [x0, x1, y0, y1] = worldView();
  const ppw = Math.min(width / (x1 - x0), height / (y1 - y0));
  return { ppw, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
}

function fitCanvas(canvas, maxWidth) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const scale = Math.min(dpr, maxWidth / Math.max(1, rect.width));
  const w = Math.max(2, Math.round(rect.width * scale)), h = Math.max(2, Math.round(rect.height * scale));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return [w, h];
}

const RENDER_WGSL = `
struct View { scale: vec2<f32>, offset: vec2<f32>, px: vec2<f32>, ppw: f32, pad: f32 };
@group(0) @binding(0) var<storage, read> feeds: array<f32>;
@group(0) @binding(1) var<storage, read> outs: array<f32>;
@group(0) @binding(2) var<storage, read> look: array<f32>;
@group(0) @binding(3) var<uniform> view: View;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec3<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) i: u32) -> VOut {
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  var o: VOut;
  if (feeds[__AGE__u * __N__u + i] < feeds[__TEMIT__u * __N__u + i]) {
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0); o.uv = vec2<f32>(0.0); o.color = vec3<f32>(0.0);
    return o;
  }
  let p = vec2<f32>(feeds[__X__u * __N__u + i], feeds[__Y__u * __N__u + i]);
  let r = max(look[i] * view.ppw, 1.2);
  let c = corners[vi];
  o.pos = vec4<f32>(p * view.scale + view.offset + c * r * view.px, 0.0, 1.0);
  o.uv = c;
  o.color = vec3<f32>(outs[__CR__u * __N__u + i], outs[__CG__u * __N__u + i], outs[__CB__u * __N__u + i]);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  if (dot(in.uv, in.uv) > 1.0) { discard; }
  return vec4<f32>(in.color, 1.0);
}`;

class GpuEngine {
  kind = "webgpu-compute";
  async init(rec, manifest, stage) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    this.device = await adapter.requestDevice();
    this.device.addEventListener?.("uncapturederror", (event) => { app.failure = String(event.error && event.error.message || event.error); });
    this.adapterInfo = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.device].filter(Boolean).join("/") : "";
    this.rec = rec; this.n = rec.n; this.abi = rec.wgsl_abi;
    this.A = this.abi.arguments; this.O = this.abi.outputs;
    const code = await fetch(rec.wgsl).then((r) => r.text());
    const module = this.device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error("generated WGSL rejected: " + errors.map((m) => `${m.lineNum}: ${m.message}`).join(" | "));
    this.compute = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.feeds = this.device.createBuffer({ size: this.A.length * this.n * 4, usage: S });
    this.outs = this.device.createBuffer({ size: this.O.length * this.n * 4, usage: S });
    this.look = this.device.createBuffer({ size: this.n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.view = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.group = this.device.createBindGroup({ layout: this.compute.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.feeds } }, { binding: 1, resource: { buffer: this.outs } }] });
    this.canvas = document.createElement("canvas");
    stage.prepend(this.canvas);
    this.ctx = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    const slot = (name) => this.A.indexOf(name), oslot = (name) => this.O.indexOf(name);
    const src = RENDER_WGSL.replaceAll("__N__", String(this.n))
      .replace(/__AGE__/g, slot("age")).replace(/__TEMIT__/g, slot("t_emit")).replace(/__X__/g, slot("x")).replace(/__Y__/g, slot("y"))
      .replace(/__CR__/g, oslot("cr")).replace(/__CG__/g, oslot("cg")).replace(/__CB__/g, oslot("cb"));
    const render = this.device.createShaderModule({ code: src });
    const rinfo = await render.getCompilationInfo();
    const rerr = rinfo.messages.filter((m) => m.type === "error");
    if (rerr.length) throw new Error("render WGSL rejected: " + rerr.map((m) => `${m.lineNum}: ${m.message}`).join(" | "));
    this.pipeline = await this.device.createRenderPipelineAsync({ layout: "auto",
      vertex: { module: render, entryPoint: "vs" }, fragment: { module: render, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" } });
    this.renderGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.feeds } }, { binding: 1, resource: { buffer: this.outs } },
      { binding: 2, resource: { buffer: this.look } }, { binding: 3, resource: { buffer: this.view } }] });
    this.host = new Float32Array(this.A.length * this.n);
    this.stateSlots = this.abi.outputs.filter((name) => name.endsWith("_next")).map((name) => [oslot(name), slot(name.slice(0, -5))]);
  }

  slot(name) { return this.A.indexOf(name); }

  reset(layout, uniforms, colours) {
    const n = this.n, host = this.host, [nx, ny] = app.manifest.nozzle;
    host.fill(0);
    const put = (name, values) => host.set(values, this.slot(name) * n);
    host.fill(nx, this.slot("x") * n, this.slot("x") * n + n);
    host.fill(ny, this.slot("y") * n, this.slot("y") * n + n);
    put("vx", layout.vx); put("vy", layout.vy);
    for (const name of ["t_emit", "sx", "sy", "mass", "drag"]) put(name, layout[name]);
    if (colours) { put("ir", colours.ir); put("ig", colours.ig); put("ib", colours.ib); }
    for (const [name, value] of Object.entries(uniforms)) host[this.slot(name) * n] = value;
    this.device.queue.writeBuffer(this.feeds, 0, host);
    this.device.queue.writeBuffer(this.outs, 0, new Float32Array(this.O.length * n));
    this.device.queue.writeBuffer(this.look, 0, layout.radius);
  }

  setUniform(name, value) { this.device.queue.writeBuffer(this.feeds, this.slot(name) * this.n * 4, new Float32Array([value])); }

  step(count) {
    const encoder = this.device.createCommandEncoder();
    for (let k = 0; k < count; k++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.compute); pass.setBindGroup(0, this.group);
      pass.dispatchWorkgroups(this.abi.groups[0], this.abi.groups[1], this.abi.groups[2]);
      pass.end();
      // the state outputs become the next step's feeds, on the GPU
      for (const [from, to] of this.stateSlots) encoder.copyBufferToBuffer(this.outs, from * this.n * 4, this.feeds, to * this.n * 4, this.n * 4);
    }
    this.device.queue.submit([encoder.finish()]);
  }

  draw() {
    const [w, h] = fitCanvas(this.canvas, 2048);
    const { ppw, cx, cy } = fit(w, h);
    const view = new Float32Array([2 * ppw / w, 2 * ppw / h, -cx * 2 * ppw / w, -cy * 2 * ppw / h, 2 / w, 2 / h, ppw, 0]);
    this.device.queue.writeBuffer(this.view, 0, view);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0.043, g: 0.051, b: 0.07, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.renderGroup);
    pass.draw(6, this.n);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // test/diagnostic only: copies the resident buffers back (the running page never does)
  async readback() {
    const bytes = (this.A.length + this.O.length) * this.n * 4;
    const read = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.feeds, 0, read, 0, this.A.length * this.n * 4);
    encoder.copyBufferToBuffer(this.outs, 0, read, this.A.length * this.n * 4, this.O.length * this.n * 4);
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
    const n = this.n, out = {};
    this.A.forEach((name, k) => { out[name] = all.subarray(k * n, (k + 1) * n); });
    this.O.forEach((name, k) => { out[name] = all.subarray((this.A.length + k) * n, (this.A.length + k + 1) * n); });
    return out;
  }

  destroy() { this.canvas.remove(); this.feeds.destroy(); this.outs.destroy(); this.look.destroy(); this.device.destroy(); }
}

class WasmEngine {
  kind = "wasm";
  async init(rec, manifest, stage) {
    this.rec = rec; this.n = rec.n;
    this.program = await TuringGraphProgram.load(rec.wasm, rec.wasm_abi);
    this.abi = this.program.abi;
    this.canvas = document.createElement("canvas");
    stage.prepend(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.inputs = {}; this.nexts = {};
    for (const name of this.abi.argument_names) this.inputs[name] = this.program.view(this.abi.inputs[name]);
    for (const out of this.abi.outputs) this.nexts[out.name] = this.program.view(out.id);
    this.state = manifest.law.state;
    this.look = new Float32Array(this.n);
    this.frame = null;
  }

  reset(layout, uniforms, colours) {
    const n = this.n, [nx, ny] = app.manifest.nozzle, I = this.inputs;
    for (const name of this.abi.argument_names) if (I[name].length > 1) I[name].fill(0);
    I.x.fill(nx); I.y.fill(ny);
    I.vx.set(layout.vx); I.vy.set(layout.vy);
    for (const name of ["t_emit", "sx", "sy", "mass", "drag"]) I[name].set(layout[name]);
    if (colours) { I.ir.set(colours.ir); I.ig.set(colours.ig); I.ib.set(colours.ib); }
    for (const [name, value] of Object.entries(uniforms)) I[name][0] = value;
    this.look.set(layout.radius);
    void n;
  }

  setUniform(name, value) { this.inputs[name][0] = value; }
  setColours(c) { this.inputs.ir.set(c.ir); this.inputs.ig.set(c.ig); this.inputs.ib.set(c.ib); }

  step(count) {
    for (let k = 0; k < count; k++) {
      this.program.run();
      for (const name of this.state) this.inputs[name].set(this.nexts[name + "_next"]);
    }
  }

  draw() {
    const [w, h] = fitCanvas(this.canvas, 900);
    if (!this.frame || this.frame.width !== w || this.frame.height !== h) this.frame = this.ctx.createImageData(w, h);
    const px = new Uint32Array(this.frame.data.buffer);
    px.fill(0xff120d0b);
    const { ppw, cx, cy } = fit(w, h);
    const I = this.inputs, O = this.nexts, n = this.n, look = this.look;
    for (let i = 0; i < n; i++) {
      if (I.age[i] < I.t_emit[i]) continue;
      const x = Math.round((I.x[i] - cx) * ppw + w / 2), y = Math.round(h / 2 - (I.y[i] - cy) * ppw);
      const r = Math.max(1, Math.round(Math.max(look[i] * ppw, 1.2)));
      const color = 0xff000000 | ((O.cb[i] * 255 + 0.5) << 16) | ((O.cg[i] * 255 + 0.5) << 8) | ((O.cr[i] * 255 + 0.5) | 0);
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r), y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
      for (let yy = y0; yy <= y1; yy++) { const row = yy * w; for (let xx = x0; xx <= x1; xx++) px[row + xx] = color; }
    }
    this.ctx.putImageData(this.frame, 0, 0);
  }

  async readback() {
    const out = {};
    for (const name of Object.keys(this.inputs)) out[name] = Float32Array.from(this.inputs[name]);
    for (const name of Object.keys(this.nexts)) out[name] = Float32Array.from(this.nexts[name]);
    return out;
  }

  destroy() { this.canvas.remove(); }
}

// ------------------------------------------------------------------- route
function decide(rec) {
  // The compiler decides (rec.route / rec.verdict); the browser can only veto.
  const v = rec.verdict;
  const detail = `n=${rec.n}: predicted ${v.speedup.toFixed(2)}x, needs ${app.manifest.cost_model.min_speedup.toFixed(1)}x`;
  if (rec.route === "webgpu-compute") {
    if (navigator.gpu) return { route: "webgpu-compute", warn: false, text: `compiler verdict: a compute dispatch pays (${detail}) - running the generated WGSL on the GPU` };
    return { route: "wasm", warn: true, text: `FALLBACK to wasm: the compiler chose WebGPU compute (${detail}) but navigator.gpu is absent here (WebGPU needs an https origin and a capable browser)` };
  }
  return { route: "wasm", warn: false, text: `compiler verdict: a dispatch does not pay (${detail}) - running the wasm product` };
}

async function buildEngine() {
  if (app.engine) { app.engine.destroy(); app.engine = null; }
  const stage = $("stage");
  const choice = decide(app.rec);
  let engine = null, reason = choice.text, warn = choice.warn;
  if (choice.route === "webgpu-compute") {
    try { engine = new GpuEngine(); await engine.init(app.rec, app.manifest, stage); }
    catch (error) {
      if (engine && engine.canvas) engine.canvas.remove();
      engine = null; warn = true;
      reason = `FALLBACK to wasm: WebGPU could not run the generated shader (${error.message || error})`;
    }
  }
  if (!engine) { engine = new WasmEngine(); await engine.init(app.rec, app.manifest, stage); }
  app.engine = engine;
  const adapter = engine.adapterInfo ? ` [${engine.adapterInfo}]` : "";
  $("route").innerHTML = `<b>${engine.kind}</b>${adapter} &middot; ${app.rec.n} ids &middot; ${reason}`;
  $("route").classList.toggle("warn", warn);
  app.route = engine.kind;
}

// -------------------------------------------------------------------- game
function uniformsNow() {
  return { ...app.rec.uniforms, mode: app.mode, use_image: app.useImage };
}

async function begin(phase) {
  app.phase = phase;
  app.useImage = phase === "replay" ? 1 : 0;
  app.steps = 0;
  app.engine.reset(app.layout, uniformsNow(), app.colours);
  app.last = performance.now();
  app.running = true;
  say();
}

async function relayout() {
  app.layout = await fetchLayout();
  if (app.raster) mapImage();
}

async function start() {
  await act({ type: "fountain_start", seed: app.seed, mix: app.mix, mode: app.mode });
  await relayout();
  await begin("fill");
}

function say(extra) {
  const t = app.steps * DT, span = app.manifest.settle_seconds;
  const title = { idle: "ready", fill: "phase 1: filling (hue from angle)", replay: "phase 2: the same fountain, coloured by the image" }[app.phase];
  const done = t >= span ? " - settled" : ` - ${Math.round(100 * Math.min(1, t / span))}%`;
  $("status").textContent = `${title}${app.phase === "idle" ? "" : done} | t ${t.toFixed(1)} s | ${app.fps.toFixed(0)} fps${extra ? " | " + extra : ""}${app.failure ? " | GPU error: " + app.failure : ""}`;
}

function frame(now) {
  requestAnimationFrame(frame);
  if (!app.engine || !app.running) return;
  const elapsed = Math.min(0.1, (now - app.last) / 1000);
  app.last = now;
  app.frames++;
  if (now - app.fpsAt > 500) { app.fps = app.frames * 1000 / (now - app.fpsAt); app.frames = 0; app.fpsAt = now; say(); }
  const done = app.steps * DT >= app.manifest.settle_seconds;
  const want = done ? 0 : Math.max(1, Math.min(16, Math.round(elapsed / DT * app.speed)));
  if (want) { app.engine.step(want); app.steps += want; }
  app.engine.draw();
}

async function pickImage(file) {
  if (!file) return;
  app.raster = await rasterize(file);
  mapImage();
  say("image mapped to " + app.rec.n + " ids");
}

async function chooseSize(tag) {
  app.tag = tag; app.rec = app.manifest.sizes[tag];
  app.running = false;
  app.raster = null; app.colours = null; $("thumb").style.display = "none"; $("replay").disabled = true;
  await buildEngine();
  await start();
}

async function boot() {
  app.manifest = await fetch("fountain.manifest.json").then((r) => r.json());
  await api("/api/new", { game: "fountain" });
  const select = $("size");
  select.innerHTML = Object.entries(app.manifest.sizes).map(([tag, rec]) => `<option value="${tag}">${rec.n.toLocaleString()} ids (${rec.route})</option>`).join("");
  app.rec = app.manifest.sizes[app.tag];
  $("start").onclick = () => start();
  $("replay").onclick = () => begin("replay");
  $("image").onchange = (e) => pickImage(e.target.files[0]);
  $("camera").onchange = (e) => pickImage(e.target.files[0]);
  $("mix").oninput = (e) => { $("mixv").textContent = Number(e.target.value).toFixed(2); };
  $("mix").onchange = async (e) => { app.mix = Number(e.target.value); await start(); };
  $("mode").onchange = (e) => { app.mode = Number(e.target.value); app.engine.setUniform("mode", app.mode); };
  $("speed").onchange = (e) => { app.speed = Number(e.target.value); };
  select.onchange = (e) => chooseSize(e.target.value);
  $("verify").onclick = async () => {
    say("asking the server to run the law natively (the first call compiles it)...");
    const reply = await act({ type: "fountain_verify", steps: Math.max(1, Math.min(app.steps, 1500)), image: app.phase === "replay" });
    const v = reply.verified;
    say(v ? `server (${v.lane}) ran ${v.steps} steps: ${v.landed} landed, digest ${v.digest}` : "server: " + (reply.error || "no reply"));
  };
  await buildEngine();
  // test and diagnostics hooks: the page itself never reads the GPU back
  window.fountainDebug = {
    app, sampleRaster,
    advance(count) { app.engine.step(count); app.steps += count; return app.steps; },
    readback: () => app.engine.readback(),
    begin, start, pickImage, mapImage,
    useRaster(raster) { app.raster = raster; mapImage(); },
  };
  window.addEventListener("resize", () => { if (app.engine) app.engine.draw(); });
  requestAnimationFrame(frame);
  await start();
}

boot().catch((error) => { $("status").textContent = "failed: " + (error.message || error); console.error(error); });
