// Page loader for a WebGPU TENSOR-LANE product (turing.wgsl-tensor-lane-abi.v1).
//
// The compiler (webgpu.emit_tensor_lane_module + emit_tensor_lane_abi) writes `<name>.wgsl` and `<name>.wgsl.abi.json`.  This
// module reads ONLY that document: it packs feeds and unpacks outputs from the declared slots, creates the device the way
// compute.js / mathlab_wgsl_timing.js do (limits checked against the declared sizes BEFORE allocating, requiredLimits no larger
// than the adapter reports, the same 2-D workgroup grid as planWgsl) and knows nothing about what the program computes.
//
// Layout it follows (all f32; `count` is baked into the shader, not a runtime value):
//   one invocation per leading-axis element.  Feed slot s (packed layout: binding bindings.feeds.binding, element
//   offset_elements; separate layout: its own binding) holds `count` floats.  A LANE slot's element i is lane i of component
//   `component` of the formal: component = row-major flat index into shape[1:], so a caller's row-major [count, ...] array
//   a gives slot[i] = a[i * comps + component], comps = product(shape[1:]).  A UNIFORM slot reads element 0 only:
//   slot[0] = a[component], a the formal's row-major flat array (a number for a rank-0 formal).  Output slot s is `count`
//   floats at offset_elements of the outputs binding; output `name` has one slot per component (row-major over shape[1:]).
//
//   const program = await loadTensorLaneProgram('/shaders/lens.wgsl.abi.json');
//   const out = await program.run({P: Float32Array(count*2), F: [x, y], H: Float32Array(count), k: 1.5, R: 2, d: 0.3});
//   out.Q -> [Float32Array(count) /*component 0*/, Float32Array(count) /*component 1*/];  interleave(out.Q) -> row-major.
const U = {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128};
const usage = (name) => (typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage[name] : U[name]);
const COMPUTE_STAGE = typeof GPUShaderStage !== 'undefined' ? GPUShaderStage.COMPUTE : 4;
const MAP_READ_MODE = typeof GPUMapMode !== 'undefined' ? GPUMapMode.READ : 1;
export const SCHEMA = 'turing.wgsl-tensor-lane-abi.v1';

const product = (shape) => shape.reduce((a, b) => a * b, 1);

// Pure: ABI + limits -> the plan (bindings, buffer sizes, packing tables, dispatch grid).  Throws Error(reason).
export function planTensorLane(abi, limits = {}) {
  if (abi.schema !== SCHEMA) throw new Error(`not a ${SCHEMA} document (schema ${JSON.stringify(abi.schema)})`);
  const need = ['entry', 'workgroup_size', 'count', 'scalar', 'bindings', 'feed_slots', 'output_slots', 'output_elements'];
  for (const key of need) if (abi[key] === undefined) throw new Error(`lane ABI lacks "${key}"`);
  if (abi.scalar !== 'f32') throw new Error(`scalar type ${abi.scalar} is not f32; this reader packs f32 only`);
  const count = abi.count, local = abi.workgroup_size[0];
  if (abi.workgroup_size[1] !== 1 || abi.workgroup_size[2] !== 1) throw new Error('only 1-D workgroups are declared by this layout reader');
  const maxBinding = limits.maxStorageBufferBindingSize ?? 134217728;
  const maxBuffer = limits.maxBufferSize ?? 268435456;
  const maxGroups = limits.maxComputeWorkgroupsPerDimension ?? 65535;
  if (local > (limits.maxComputeWorkgroupSizeX ?? 256)) throw new Error(`workgroup size ${local} exceeds the device limit`);
  const packed = abi.bindings.feed_layout === 'packed';
  if (!packed && abi.bindings.feed_layout !== 'separate') throw new Error(`unknown feed layout ${abi.bindings.feed_layout}`);
  const bindings = new Map();
  const bind = (binding, elements, access) => {
    const bytes = Math.max(elements, 1) * 4;
    if (bindings.has(binding)) throw new Error(`binding ${binding} declared twice`);
    bindings.set(binding, {binding, bytes, access});
  };
  if (packed) bind(abi.bindings.feeds.binding, abi.feed_elements, 'read');
  else for (const slot of abi.feed_slots) bind(slot.binding, slot.offset_elements + count, 'read');
  bind(abi.bindings.outputs.binding, abi.output_elements, 'read_write');
  if ((limits.maxStorageBuffersPerShaderStage ?? 8) < bindings.size) throw new Error('device has too few storage bindings per stage');
  for (const b of bindings.values()) {
    if (b.bytes > maxBinding || b.bytes > maxBuffer) {
      throw new Error(`binding ${b.binding} needs ${b.bytes} bytes, over the device storage binding limit ${Math.min(maxBinding, maxBuffer)}; ` +
        'rows cannot be re-chunked (count is baked into the shader)');
    }
  }
  // feeds grouped by formal name, in slot order
  const formals = new Map();
  for (const slot of abi.feed_slots) {
    if (!formals.has(slot.name)) formals.set(slot.name, {name: slot.name, shape: slot.shape, uniform: slot.uniform, slots: []});
    const formal = formals.get(slot.name);
    if (formal.uniform !== slot.uniform) throw new Error(`formal "${slot.name}" mixes lane and uniform slots`);
    formal.slots.push(slot);
  }
  for (const formal of formals.values()) {
    formal.comps = formal.uniform ? product(formal.shape) : product(formal.shape.slice(1));
    formal.length = formal.uniform ? formal.comps : count * formal.comps;
    if (formal.slots.length !== formal.comps) throw new Error(`formal "${formal.name}" declares ${formal.slots.length} slots for ${formal.comps} components`);
  }
  const outputs = new Map();
  for (const slot of abi.output_slots) {
    if (!outputs.has(slot.name)) outputs.set(slot.name, {name: slot.name, shape: slot.shape, slots: []});
    outputs.get(slot.name).slots.push(slot);
  }
  const totalGroups = Math.ceil(count / local);
  const gx = Math.min(totalGroups, maxGroups), gy = Math.ceil(totalGroups / gx);
  if (gy > maxGroups) throw new Error(`${count} rows need a ${gx}x${gy} workgroup grid, over the device limit ${maxGroups}`);
  return {count, entry: abi.entry, bindings: [...bindings.values()].sort((a, b) => a.binding - b.binding),
          outputBinding: abi.bindings.outputs.binding, outputBytes: abi.output_elements * 4,
          feedBuffers: packed ? [{binding: abi.bindings.feeds.binding, elements: abi.feed_elements}] :
            abi.feed_slots.map((slot) => ({binding: slot.binding, elements: slot.offset_elements + count})),
          formals: [...formals.values()], outputs: [...outputs.values()], groups: [gx, gy, 1], workgroup_size: abi.workgroup_size};
}

// Pure: caller values -> Map(binding -> Float32Array) packed strictly by the declared slots.  Values: number | array-like.
export function packTensorLaneFeeds(plan, values) {
  const known = new Set(plan.formals.map((f) => f.name));
  for (const key of Object.keys(values)) if (!known.has(key)) throw new Error(`no feed "${key}" is declared (declared: ${[...known].join(', ')})`);
  const buffers = new Map(plan.feedBuffers.map((b) => [b.binding, new Float32Array(b.elements)]));
  for (const formal of plan.formals) {
    if (!(formal.name in values)) throw new Error(`feed "${formal.name}" was not supplied`);
    const raw = values[formal.name];
    const source = typeof raw === 'number' ? [raw] : raw;
    if (!source || source.length !== formal.length) {
      throw new Error(`feed "${formal.name}" needs ${formal.length} values (shape ${JSON.stringify(formal.shape)}${formal.uniform ? ', uniform' : `, ${plan.count} lanes x ${formal.comps}`}), got ${source ? source.length : 'none'}`);
    }
    for (const slot of formal.slots) {
      const target = buffers.get(slot.binding);
      if (formal.uniform) target[slot.offset_elements] = source[slot.component];
      else for (let i = 0; i < plan.count; i++) target[slot.offset_elements + i] = source[i * formal.comps + slot.component];
    }
  }
  return buffers;
}

// Pure: the read-back outputs buffer -> {name: [Float32Array(count) per component]}.
export function unpackTensorLaneOutputs(plan, data) {
  const result = {};
  for (const output of plan.outputs) {
    result[output.name] = output.slots.map((slot) => data.slice(slot.offset_elements, slot.offset_elements + slot.length_elements));
  }
  return result;
}

// [Float32Array per component] -> one row-major Float32Array [count, components].
export function interleave(components) {
  const n = components[0].length, k = components.length, out = new Float32Array(n * k);
  for (let c = 0; c < k; c++) for (let i = 0; i < n; i++) out[i * k + c] = components[c][i];
  return out;
}

async function sha256Hex(text) {
  if (typeof crypto === 'undefined' || !crypto.subtle) return null;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// url: the `.wgsl.abi.json` document (the shader is fetched from abi.shader, relative to it).
// options: {device} to share a device; {shader} to supply the shader text.
export async function loadTensorLaneProgram(url, {device = null, shader = null, fetchImpl = (typeof fetch !== 'undefined' ? fetch : null)} = {}) {
  const get = async (target, kind) => {
    const response = await fetchImpl(target);
    if (!response.ok) throw new Error(`${target}: HTTP ${response.status}`);
    return kind === 'json' ? response.json() : response.text();
  };
  const abi = await get(url, 'json');
  const code = shader ?? await get(new URL(abi.shader, new URL(url, typeof location !== 'undefined' ? location.href : undefined)).href, 'text');
  const digest = await sha256Hex(code);
  if (digest !== null && abi.shader_sha256 && digest !== abi.shader_sha256) throw new Error(`${abi.shader}: content hash ${digest} does not match the ABI's ${abi.shader_sha256}`);
  let lost = null;
  let plan;
  if (!device) {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('no WebGPU in this browser (navigator.gpu is absent)');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('WebGPU is present but no adapter was returned');
    const L = adapter.limits;
    plan = planTensorLane(abi, L);
    const ask = (name, want) => Math.min(want, L[name]);   // request no more than the adapter reports
    const bytes = Math.max(...plan.bindings.map((b) => b.bytes));
    device = await adapter.requestDevice({requiredLimits: {
      maxStorageBufferBindingSize: ask('maxStorageBufferBindingSize', Math.max(bytes, 134217728)),
      maxBufferSize: ask('maxBufferSize', Math.max(bytes, 268435456)),
      maxStorageBuffersPerShaderStage: ask('maxStorageBuffersPerShaderStage', Math.max(plan.bindings.length, 8))}});
  } else {
    plan = planTensorLane(abi, device.limits || {});
  }
  if (device.lost) device.lost.then((info) => { lost = info; });

  device.pushErrorScope('validation');
  const module = device.createShaderModule({code});
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = (info.messages || []).filter((m) => m.type === 'error');
    if (errors.length) { await device.popErrorScope(); throw new Error('WGSL compile error: ' + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')); }
  }
  const bindGroupLayout = device.createBindGroupLayout({entries: plan.bindings.map((b) => ({
    binding: b.binding, visibility: COMPUTE_STAGE, buffer: {type: b.access === 'read' ? 'read-only-storage' : 'storage'}}))});
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    compute: {module, entryPoint: plan.entry}});
  const buffers = new Map(plan.bindings.map((b) => [b.binding, device.createBuffer({
    size: b.bytes, usage: usage('STORAGE') | usage('COPY_DST') | usage('COPY_SRC')})]));
  const staging = device.createBuffer({size: plan.outputBytes, usage: usage('MAP_READ') | usage('COPY_DST')});
  const bindGroup = device.createBindGroup({layout: bindGroupLayout, entries: plan.bindings.map((b) => ({binding: b.binding, resource: {buffer: buffers.get(b.binding)}}))});
  const setup = await device.popErrorScope();
  if (setup) throw new Error('WebGPU validation error: ' + (setup.message || setup));

  let queue = Promise.resolve();
  const runOnce = async (values) => {
    const packed = packTensorLaneFeeds(plan, values);
    for (const [binding, data] of packed) device.queue.writeBuffer(buffers.get(binding), 0, data);
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(plan.groups[0], plan.groups[1], plan.groups[2]);
    pass.end();
    encoder.copyBufferToBuffer(buffers.get(plan.outputBinding), 0, staging, 0, plan.outputBytes);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error) throw new Error('WebGPU validation error: ' + (error.message || error));
    await staging.mapAsync(MAP_READ_MODE);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    if (lost) throw new Error('WebGPU device lost: ' + (lost.message || lost.reason));
    return unpackTensorLaneOutputs(plan, data);
  };
  return {
    abi, plan, device,
    run: (values) => { const next = queue.then(() => runOnce(values)); queue = next.catch(() => {}); return next; },
    destroy: () => { for (const b of buffers.values()) b.destroy(); staging.destroy(); },
  };
}
