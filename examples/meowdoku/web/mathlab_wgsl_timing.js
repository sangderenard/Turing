// Browser timing of a math-lab WGSL product.  The binding/buffer/dispatch plan is read from the product's DECLARED layout
// (law.wgsl.abi.json, written by math_lab.py from WGSLPassProgram): lane bindings 0..bindings-1 (group 0), the uniform
// `head` at binding `bindings`, slot s in lane binding floor(s / slots_per_binding) at element (s % slots_per_binding) * count,
// one compute entry point per dispatch run in order inside one compute pass, 1-D workgroups laid out on a 2-D grid
// (i = gid.x + gid.y * num_workgroups.x * workgroup_size.x).  The shader text is only handed to createShaderModule.
// `count` is baked into the shader (the slot stride and the bounds guard), so rows == abi.count: rows cannot be re-chunked;
// only the workgroup grid is re-shaped to the device's per-dimension limit, and a product whose buffers exceed the
// device's binding limit is refused with a reason.
const U = {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128};
const usage = (name) => (typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage[name] : U[name]);
const COMPUTE_STAGE = typeof GPUShaderStage !== 'undefined' ? GPUShaderStage.COMPUTE : 4;
const MAP_READ_MODE = typeof GPUMapMode !== 'undefined' ? GPUMapMode.READ : 1;

const median = (values) => { const s = [...values].sort((x, y) => x - y); return s[s.length >> 1]; };

// Pure: declared layout + limits -> the plan.  Throws Error(reason) when the product cannot run on these limits.
export function planWgsl(abi, limits = {}) {
  const need = ['count', 'workgroup_size', 'entries', 'slots', 'bindings', 'slots_per_binding', 'feed_slots', 'output_slots', 'arguments', 'outputs'];
  for (const key of need) if (abi[key] === undefined) throw new Error(`product layout lacks "${key}"`);
  const count = abi.count, spb = abi.slots_per_binding, local = abi.workgroup_size[0];
  const maxBinding = limits.maxStorageBufferBindingSize ?? 134217728;
  const maxBuffer = limits.maxBufferSize ?? 268435456;
  const maxGroups = limits.maxComputeWorkgroupsPerDimension ?? 65535;
  if (abi.workgroup_size[1] !== 1 || abi.workgroup_size[2] !== 1) throw new Error('only 1-D workgroups are declared by this layout reader');
  if (local > (limits.maxComputeWorkgroupSizeX ?? 256)) throw new Error(`workgroup size ${local} exceeds the device limit`);
  if ((limits.maxStorageBuffersPerShaderStage ?? 8) < abi.bindings + 1) throw new Error('device has too few storage bindings per stage');
  const uniforms = abi.arena_elements - abi.slots * count;
  if (uniforms !== 0) throw new Error(`product declares ${uniforms} uniform feed(s) in the head binding; the page has no values for them`);
  const lanes = [];
  for (let b = 0; b < abi.bindings; b++) {
    const slots = Math.min(spb, abi.slots - b * spb);
    const bytes = slots * count * 4;
    if (slots <= 0) throw new Error(`lane binding ${b} holds no slots`);
    if (bytes > maxBinding || bytes > maxBuffer) {
      throw new Error(`lane binding ${b} needs ${bytes} bytes, over the device storage binding limit ${Math.min(maxBinding, maxBuffer)}; ` +
        'rows cannot be re-chunked (count is baked into the shader)');
    }
    lanes.push({group: 0, binding: b, bytes, slots});
  }
  const head = {group: 0, binding: abi.bindings, bytes: 4};
  const place = (slot) => ({binding: Math.floor(slot / spb), offset: (slot % spb) * count * 4, bytes: count * 4});
  const inputs = abi.arguments.map((name) => {
    if (!(name in abi.feed_slots)) throw new Error(`input "${name}" has no declared feed slot`);
    return {name, ...place(abi.feed_slots[name])};
  });
  const outputs = abi.outputs.map((name) => {
    if (!(name in abi.output_slots)) throw new Error(`output "${name}" has no declared output slot`);
    return {name, ...place(abi.output_slots[name])};
  });
  const totalGroups = Math.ceil(count / local);
  const gx = Math.min(totalGroups, maxGroups), gy = Math.ceil(totalGroups / gx);
  if (gy > maxGroups) throw new Error(`${count} rows need a ${gx}x${gy} workgroup grid, over the device limit ${maxGroups}`);
  const dispatches = abi.entries.map((entry) => ({entry, groups: [gx, gy, 1]}));
  return {count, lanes, head, inputs, outputs, dispatches, workgroup_size: abi.workgroup_size};
}

// product: {shader: <law.wgsl text>, abi: <law.wgsl.abi.json object>}.  columns: {name: array-like} (cycled to abi.count rows).
// options: {device, reference: {rows, inputs: {name: array}, outputs: {name: array}} | null (its inputs overwrite the first rows)}.
export async function timeWgsl(product, columns, rows, {device, reference = null, now = () => performance.now()} = {}) {
  let lost = null;
  let scopeOpen = false;
  try {
    let adapterInfo = {};
    if (!device) {
      if (typeof navigator === 'undefined' || !navigator.gpu) return {ok: false, reason: 'no WebGPU in this browser (navigator.gpu is absent)'};
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return {ok: false, reason: 'WebGPU is present but no adapter was returned'};
      const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}) || {};
      adapterInfo = {vendor: info.vendor || '', architecture: info.architecture || ''};
      device = await adapter.requestDevice();
    } else if (device.adapterInfo) {
      adapterInfo = {vendor: device.adapterInfo.vendor || '', architecture: device.adapterInfo.architecture || ''};
    }
    if (device.lost) device.lost.then((info) => { lost = info; });
    const abi = product.abi;
    let plan;
    try { plan = planWgsl(abi, device.limits || {}); } catch (error) { return {ok: false, reason: error.message}; }
    const n = plan.count;
    const features = new Map();
    for (const input of plan.inputs) {
      const source = columns[input.name];
      if (!source || !source.length) return {ok: false, reason: `no column supplied for input "${input.name}"`};
      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = source[i % source.length];
      features.set(input.name, data);
    }
    let checked = 0;
    if (reference && reference.rows > 0 && reference.inputs) {
      checked = Math.min(reference.rows, n);
      for (const input of plan.inputs) {
        if (reference.inputs[input.name]) features.get(input.name).set(Float32Array.from(reference.inputs[input.name]).subarray(0, checked));
      }
    }

    const guarded = async (work) => {
      device.pushErrorScope('validation');
      scopeOpen = true;
      let value, thrown = null;
      try { value = await work(); } catch (error) { thrown = error; }
      const error = await device.popErrorScope();
      scopeOpen = false;
      if (error) throw new Error('WebGPU validation error: ' + (error.message || error));
      if (thrown) throw thrown;
      return value;
    };

    // compile: module, explicit bind group layout from the declared bindings, one pipeline per declared entry
    const t0 = now();
    const built = await guarded(async () => {
      const module = device.createShaderModule({code: product.shader});
      if (module.getCompilationInfo) {
        const info = await module.getCompilationInfo();
        const errors = (info.messages || []).filter((m) => m.type === 'error');
        if (errors.length) throw new Error('WGSL compile error: ' + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
      }
      const entries = [...plan.lanes, plan.head].map((b) => ({binding: b.binding, visibility: COMPUTE_STAGE, buffer: {type: 'storage'}}));
      const bindGroupLayout = device.createBindGroupLayout({entries});
      const layout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
      const pipelines = [];
      for (const dispatch of plan.dispatches) {
        pipelines.push(await device.createComputePipelineAsync({layout, compute: {module, entryPoint: dispatch.entry}}));
      }
      return {bindGroupLayout, pipelines};
    });
    const compile_ms = now() - t0;

    const buffers = [];
    const bindGroup = await guarded(async () => {
      for (const lane of plan.lanes) buffers.push(device.createBuffer({size: lane.bytes, usage: usage('STORAGE') | usage('COPY_DST') | usage('COPY_SRC')}));
      buffers.push(device.createBuffer({size: plan.head.bytes, usage: usage('STORAGE') | usage('COPY_DST')}));
      return device.createBindGroup({layout: built.bindGroupLayout, entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
    });
    const staging = plan.outputs.map((o) => device.createBuffer({size: o.bytes, usage: usage('MAP_READ') | usage('COPY_DST')}));

    const upload = () => { for (const input of plan.inputs) device.queue.writeBuffer(buffers[input.binding], input.offset, features.get(input.name)); };
    const encodeRun = (withCopy) => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindGroup);
      plan.dispatches.forEach((dispatch, index) => {
        pass.setPipeline(built.pipelines[index]);
        pass.dispatchWorkgroups(...dispatch.groups);
      });
      pass.end();
      if (withCopy) plan.outputs.forEach((o, k) => encoder.copyBufferToBuffer(buffers[o.binding], o.offset, staging[k], 0, o.bytes));
      device.queue.submit([encoder.finish()]);
    };
    // The arena reuses slots, so inputs are re-uploaded (and the upload drained) before every run; only dispatch is timed.
    const runOnce = async (withCopy) => {
      upload();
      await device.queue.onSubmittedWorkDone();
      const start = now();
      encodeRun(withCopy);
      await device.queue.onSubmittedWorkDone();
      return now() - start;
    };

    const first_run_ms = await guarded(() => runOnce(false));
    await guarded(() => runOnce(false));
    const times = [];
    for (let i = 0; i < 5; i++) times.push(await guarded(() => runOnce(false)));
    const steady_ms = median(times);

    // read back: one more run with copies; readback_ms = map + copy-out of all outputs
    await guarded(() => runOnce(true));
    const r0 = now();
    const got = {};
    await guarded(async () => {
      for (let k = 0; k < plan.outputs.length; k++) {
        await staging[k].mapAsync(MAP_READ_MODE);
        got[plan.outputs[k].name] = new Float32Array(staging[k].getMappedRange().slice(0));
        staging[k].unmap();
      }
    });
    const readback_ms = now() - r0;
    let max_abs_err = null;
    if (reference && reference.outputs && checked > 0) {
      max_abs_err = 0;
      for (const o of plan.outputs) {
        const want = reference.outputs[o.name];
        if (!want) continue;
        for (let i = 0; i < checked; i++) max_abs_err = Math.max(max_abs_err, Math.abs(got[o.name][i] - Math.fround(want[i])));
      }
    }
    if (lost) return {ok: false, reason: `device lost: ${lost.message || lost.reason}`};
    for (const b of [...buffers, ...staging]) if (b.destroy) b.destroy();
    return {ok: true, rows: n, ns_per_row: steady_ms * 1e6 / n, compile_ms, first_run_ms, steady_ms, readback_ms, max_abs_err,
      adapter: adapterInfo, precision: 'f32'};
  } catch (error) {
    if (scopeOpen) { try { await device.popErrorScope(); } catch (ignored) { /* scope already gone */ } }
    return {ok: false, reason: lost ? `device lost: ${lost.message || lost.reason}` : (error && error.message) || String(error)};
  }
}
