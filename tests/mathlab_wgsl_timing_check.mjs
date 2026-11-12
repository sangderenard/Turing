// node tests/mathlab_wgsl_timing_check.mjs : no GPU; a recording fake device proves the plan matches the declared layout.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {planWgsl, timeWgsl} from '../examples/meowdoku/web/mathlab_wgsl_timing.js';

const sample = JSON.parse(readFileSync(new URL('./data/mathlab_wgsl_sample.json', import.meta.url), 'utf-8'));

function fakeDevice(limits, calls) {
  const rec = (name, fn = () => undefined) => (...args) => { calls.push([name, ...args]); return fn(...args); };
  let clock = 0;
  const buffer = (d) => ({d, mapAsync: async () => {}, getMappedRange: () => new Float32Array(d.size / 4).fill(2).buffer, unmap() {}, destroy() {}});
  return {
    limits, clock: () => (clock += 1),
    pushErrorScope: rec('pushErrorScope'), popErrorScope: async () => { calls.push(['popErrorScope']); return null; },
    createShaderModule: rec('createShaderModule', () => ({getCompilationInfo: async () => ({messages: []})})),
    createBindGroupLayout: rec('createBindGroupLayout', (d) => d),
    createPipelineLayout: rec('createPipelineLayout', (d) => d),
    createComputePipelineAsync: async (d) => { calls.push(['createComputePipelineAsync', d.compute.entryPoint]); return {entry: d.compute.entryPoint}; },
    createBuffer: rec('createBuffer', buffer),
    createBindGroup: rec('createBindGroup', (d) => d),
    createCommandEncoder: () => ({
      beginComputePass: () => ({setBindGroup: rec('setBindGroup'), setPipeline: rec('setPipeline'), dispatchWorkgroups: rec('dispatchWorkgroups'), end() {}}),
      copyBufferToBuffer: rec('copyBufferToBuffer'), finish: () => ({}),
    }),
    queue: {writeBuffer: rec('writeBuffer'), submit: rec('submit'), onSubmittedWorkDone: async () => {}},
  };
}
const kinds = (calls, name) => calls.filter((c) => c[0] === name);

// ---- the sample's declared layout: 1 lane binding (slots 0..2, 1024 rows), head at binding 1, one entry p0 ----
const abi = sample.abi;
let plan = planWgsl(abi, {});
assert.deepEqual(plan.lanes, [{group: 0, binding: 0, bytes: 3 * 1024 * 4, slots: 3}]);
assert.deepEqual(plan.head, {group: 0, binding: 1, bytes: 4});
assert.deepEqual(plan.inputs.map((i) => [i.name, i.binding, i.offset]), [['x', 0, 0], ['y', 0, 4096]]);
assert.deepEqual(plan.outputs.map((o) => [o.name, o.binding, o.offset]), [['r', 0, 8192]]);
assert.deepEqual(plan.dispatches, [{entry: 'p0', groups: [16, 1, 1]}]);

// ---- a full run through the fake: calls match the plan ----
{
  const calls = [];
  const device = fakeDevice({maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535}, calls);
  let t = 0;
  const result = await timeWgsl({shader: sample.shader, abi}, {x: [1, 2, 3], y: [4, 5]}, 1024, {device, now: () => (t += 1)});
  assert.equal(result.ok, true, result.reason);
  assert.equal(result.rows, 1024);
  assert.equal(result.precision, 'f32');
  assert.equal(kinds(calls, 'createShaderModule')[0][1].code, sample.shader);
  const layoutEntries = kinds(calls, 'createBindGroupLayout')[0][1].entries;
  assert.deepEqual(layoutEntries.map((e) => e.binding), [0, 1]);
  assert.deepEqual(kinds(calls, 'createComputePipelineAsync').map((c) => c[1]), ['p0']);
  const bufs = kinds(calls, 'createBuffer').map((c) => c[1].size);
  assert.deepEqual(bufs.slice(0, 2), [12288, 4]);                       // lane0, head
  assert.deepEqual(bufs.slice(2), [4096]);                              // staging for r
  assert.deepEqual(kinds(calls, 'createBindGroup')[0][1].entries.map((e) => e.binding), [0, 1]);
  assert.deepEqual(kinds(calls, 'writeBuffer').slice(0, 2).map((c) => [c[2]]), [[0], [4096]]);
  assert.deepEqual(kinds(calls, 'dispatchWorkgroups')[0].slice(1), [16, 1, 1]);
  assert.equal(kinds(calls, 'dispatchWorkgroups').length, 1 + 1 + 5 + 1);   // first, warmup, 5 steady, readback run
  assert.deepEqual(kinds(calls, 'copyBufferToBuffer')[0].slice(2, 3), [8192]);
  assert.equal(kinds(calls, 'pushErrorScope').length, kinds(calls, 'popErrorScope').length);
}

// ---- 1e5 rows, small limits: the grid is re-shaped under maxComputeWorkgroupsPerDimension, two lane bindings, two entries ----
{
  const big = {...abi, count: 100000, groups: [16384, 1, 1], slots: 5, slots_per_binding: 2, bindings: 3, arena_elements: 500000,
    entries: ['p0', 'p1'], feed_slots: {x: 0, y: 3}, output_slots: {r: 4}};
  const small = {maxStorageBufferBindingSize: 800000, maxBufferSize: 1 << 28, maxComputeWorkgroupsPerDimension: 500, maxComputeWorkgroupSizeX: 256};
  const p = planWgsl(big, small);
  assert.deepEqual(p.lanes.map((l) => [l.binding, l.bytes]), [[0, 800000], [1, 800000], [2, 400000]]);
  assert.deepEqual(p.head, {group: 0, binding: 3, bytes: 4});
  assert.deepEqual(p.inputs.map((i) => [i.name, i.binding, i.offset]), [['x', 0, 0], ['y', 1, 400000]]);
  assert.deepEqual(p.outputs.map((o) => [o.name, o.binding, o.offset]), [['r', 2, 0]]);
  const total = Math.ceil(100000 / 64);                                  // 1563 workgroups
  assert.deepEqual(p.dispatches, [{entry: 'p0', groups: [500, 4, 1]}, {entry: 'p1', groups: [500, 4, 1]}]);
  assert.ok(500 * 4 >= total && 500 * 3 < total);
  const calls = [];
  const device = fakeDevice(small, calls);
  const result = await timeWgsl({shader: sample.shader, abi: big}, {x: [1], y: [1]}, 100000, {device});
  assert.equal(result.ok, true, result.reason);
  assert.deepEqual(kinds(calls, 'dispatchWorkgroups').slice(0, 2).map((c) => c.slice(1)), [[500, 4, 1], [500, 4, 1]]);
  // a binding over the limit is refused with a reason, not chunked
  const refused = await timeWgsl({shader: sample.shader, abi: big}, {x: [1], y: [1]}, 100000,
    {device: fakeDevice({...small, maxStorageBufferBindingSize: 700000}, [])});
  assert.equal(refused.ok, false);
  assert.match(refused.reason, /storage binding limit/);
  const grid = await timeWgsl({shader: sample.shader, abi: big}, {x: [1], y: [1]}, 100000,
    {device: fakeDevice({...small, maxComputeWorkgroupsPerDimension: 20}, [])});
  assert.equal(grid.ok, false);
  assert.match(grid.reason, /workgroup grid/);
}

// ---- validation error from the error scope, and the no-WebGPU path ----
{
  const calls = [];
  const device = fakeDevice({}, calls);
  device.popErrorScope = async () => ({message: 'bad pipeline'});
  const bad = await timeWgsl({shader: sample.shader, abi}, {x: [1], y: [1]}, 1024, {device});
  assert.equal(bad.ok, false);
  assert.match(bad.reason, /validation error: bad pipeline/);
}
{
  const none = await timeWgsl({shader: sample.shader, abi}, {x: [1], y: [1]}, 1024);
  assert.equal(none.ok, false);
  assert.match(none.reason, /no WebGPU/);
}
console.log('mathlab_wgsl_timing_check: all assertions passed');
