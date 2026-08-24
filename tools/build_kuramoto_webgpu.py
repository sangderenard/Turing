"""Emit the Kuramoto field as a WebGPU page: shader, buffers, and host.

The per-cell advance is the SAME SymPy expression the CPU demos compile
-- imported from ``demo_kuramoto_field``, never restated -- materialised
into AbstractTensor Python and lowered to WGSL by the repository's own
WebGPU backend. What this file adds is only what the compiler does not
emit: a host page to run it and two small shaders that MOVE data rather
than compute with it.

Three things are worth stating plainly about the shape of the result.

WGSL has no f64. That is not a reason to give up precision, it is the
reason the float32 limb ladder exists: this emits at ``Precision[2,
float32]``, two 24-bit limbs, which carries more significand than the
binary64 it replaces rather than less. The lowering uses the SPLIT
two_product -- Veltkamp halves at 4097, exact for a 24-bit significand --
so no fused multiply-add appears anywhere and WGSL's permission to
double-round an ``fma()`` never applies.

The emitted compute ABI is elementwise: every formal is read as
``feed_N[linear_index]``, one invocation per cell, with no uniform or
scalar concept. So the coefficients are replicated to full length rather
than bound as uniforms. It is a little wasteful of memory and completely
free at runtime, and it means nothing here has to argue with the ABI.

Only the MATHEMATICS is compiler-emitted. The gather shader (which reads
each cell's four neighbours off the torus) and the present shader (which
paints phase as hue) are hand-written and live at the bottom of this
file, clearly separated, because both are data movement and neither
computes a transcendental.
"""

from __future__ import annotations

import argparse
import ast
import json
from fractions import Fraction
from pathlib import Path
import sys

import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.demo_kuramoto_field import (  # noqa: E402
    NEIGHBOURS, core_terms, kuramoto_equation,
)
from src.common.tensors.signal_symbolic import (  # noqa: E402
    constant_rational, limb_decomposition,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ir_identities import (  # noqa: E402
    inline_host_linear_source_regions, narrow_float64_to_float32,
    two_product_flavor_scope,
)
from src.compiler.ssa_python_materializer import (  # noqa: E402
    materialize_function_body,
)
from src.compiler.symbolic_equation_compiler import (  # noqa: E402
    compile_sympy_equations,
)
from src.compiler import ssa_webgpu_backend as webgpu  # noqa: E402

#: Per-cell fields the host supplies as real arrays. Everything else the
#: equation names is one value shared by every cell.
FIELDS = ("theta", "omega", *NEIGHBOURS)


def cell_source(equation, width: int, element: str) -> tuple[str, tuple]:
    """The per-cell advance as an annotated AbstractTensor function.

    No loop: a compute invocation IS the loop body, and the emitted entry
    already derives ``linear_index`` from the invocation id.
    """

    compiled = compile_sympy_equations([equation], name="kuramoto_step")
    statements, needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    if needs_math:
        raise RuntimeError("a scalar opcode reached a tensor program")

    assigned, loaded = set(), set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store)
             else loaded).add(node.id)
    parameters = tuple(sorted(loaded - assigned))

    annotation = ast.Subscript(
        value=ast.Name(id="Precision", ctx=ast.Load()),
        slice=ast.Tuple(
            elts=[ast.Constant(value=width),
                  ast.Name(id=element, ctx=ast.Load())],
            ctx=ast.Load(),
        ),
        ctx=ast.Load(),
    )
    function = ast.FunctionDef(
        name="kuramoto_step",
        args=ast.arguments(
            posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[],
            args=[
                ast.arg(arg=name, annotation=ast.fix_missing_locations(
                    ast.parse(ast.unparse(annotation), mode="eval").body
                ))
                for name in parameters
            ],
        ),
        body=statements, decorator_list=[], returns=None, type_params=[],
    )
    source = ast.unparse(ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    ))
    return source, parameters


def build(width: int, digits: int, cells: int):
    """Lower, narrow, flatten, and emit WGSL. Returns the module and plan."""

    sine = list(core_terms("sin", digits))
    cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))

    equation, constants = kuramoto_equation(terms)

    with two_product_flavor_scope("split"):
        source, parameters = cell_source(equation, width, "float32")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "kuramoto_step", name="kur",
        )
    narrowed = narrow_float64_to_float32(module)
    flat, _receipts = inline_host_linear_source_regions(module.functions)

    entry = "kur__kuramoto_step"
    function = flat[entry]
    returned = _returned_values(function)
    emitted = webgpu.emit_module(
        flat, name="kuramoto", count=cells,
        outputs={entry: returned},
    )
    return emitted, flat, entry, sine, cosine, constants, terms, narrowed


def _returned_values(function):
    for block in function.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) in ("Ret", "Return"):
                return list(instruction.args)
    return []


def feed_plan(function, entry: str, width: int, sine, cosine, constants,
              quarter_exact, coupling: float, dt: float):
    """Which SSA id each feed buffer answers to, and what goes in it.

    A ``Precision[n]`` formal became n formals, one per limb, so the plan
    is written in limb slots and not in values.
    """

    ids = dict(function.metadata["parameter_names"])
    rows = dict(function.metadata.get("precision_lowered_values") or ())
    for name, identifier in tuple(ids.items()):
        row = rows.get(int(identifier))
        if row:
            for position, limb in enumerate(tuple(row)[1:], start=1):
                ids.setdefault(f"{name}__limb{position}", int(limb))

    def limbs_of(value):
        # The element MUST be stated. Splitting for float64 limbs and then
        # consuming the parts as float32 makes the head unrepresentable and
        # the tail meaningless: the correction belongs to bits the f32 head
        # does not have.
        return [
            float(part)
            for part in limb_decomposition(value, width, element="float32")
        ]

    scalars = {
        "coupling": limbs_of(Fraction(coupling).limit_denominator(10**9)),
        "dt": limbs_of(Fraction(dt).limit_denominator(10**9)),
        "quarter": limbs_of(quarter_exact),
        "neg_quarter": limbs_of(-quarter_exact),
        "inv_quarter": limbs_of(1 / quarter_exact),
    }
    for prefix, values in (("c", sine), ("d", cosine)):
        for index, value in enumerate(values):
            scalars[f"{prefix}{index}"] = limbs_of(value)
    for name, value in constants.items():
        scalars[name] = limbs_of(value)

    # The feeds are ONE SPAN: formal at position k occupies
    # [k * count, (k + 1) * count). So the plan is written by POSITION in
    # the emitted signature, not by SSA id, and the host fills one buffer.
    by_id = {int(identifier): name for name, identifier in ids.items()}
    plan = {"slots": [], "unbound": []}
    for position, formal in enumerate(function.args):
        name = by_id.get(int(formal.id))
        if name is None:
            plan["unbound"].append(f"%{formal.id}")
            continue
        base, _, suffix = name.partition("__limb")
        limb = int(suffix) if suffix else 0
        if base in FIELDS:
            plan["slots"].append(
                {"at": position, "field": base, "limb": limb}
            )
        elif base in scalars:
            parts = scalars[base]
            plan["slots"].append({
                "at": position,
                "value": parts[limb] if limb < len(parts) else 0.0,
            })
        else:
            plan["unbound"].append(name)
    return plan, ids


# -- everything below is HAND-WRITTEN, and none of it is mathematics -----
#
# The gather shader reads each cell's four neighbours off the torus and
# the present shader paints phase as hue. Both only move or display
# values that the compiled kernel produced.

GATHER_WGSL = """
// Offsets into the one feed span, supplied explicitly. A field's limbs
// are NOT consecutive -- the emitter lays out limb 0 of every formal
// before limb 1 of any of them -- so striding by the limb index walks
// into a NEIGHBOUR's slot and then into the coefficients. Measured, that
// overwrote the series with phase data every frame and the field came
// back entirely NaN.
struct Shape { wide: u32, high: u32, count: u32, limbs: u32 };
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> feeds: array<f32>;
// [theta..., up..., down..., left..., right...], `limbs` entries each.
@group(0) @binding(2) var<storage, read> offsets: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= shape.count) { return; }
  let row = i / shape.wide;
  let col = i % shape.wide;
  let up_i = ((row + shape.high - 1u) % shape.high) * shape.wide + col;
  let down_i = ((row + 1u) % shape.high) * shape.wide + col;
  let left_i = row * shape.wide + (col + shape.wide - 1u) % shape.wide;
  let right_i = row * shape.wide + (col + 1u) % shape.wide;
  // Every limb moves together: a gather is data movement and cannot
  // round, so the expansion crosses it intact.
  for (var k: u32 = 0u; k < shape.limbs; k = k + 1u) {
    let src = offsets[k];
    feeds[offsets[shape.limbs + k] + i] = feeds[src + up_i];
    feeds[offsets[2u * shape.limbs + k] + i] = feeds[src + down_i];
    feeds[offsets[3u * shape.limbs + k] + i] = feeds[src + left_i];
    feeds[offsets[4u * shape.limbs + k] + i] = feeds[src + right_i];
  }
}
"""

PRESENT_WGSL = """
struct Shape { wide: u32, high: u32, count: u32, limbs: u32 };
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> feeds: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  var points = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let p = points[index];
  var out: VertexOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

// Phase is an angle, so it must be painted on a colour wheel: a linear
// ramp would put a seam where the phase wraps and invent a discontinuity
// the field does not have.
@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  let x = min(u32(in.uv.x * f32(shape.wide)), shape.wide - 1u);
  let y = min(u32(in.uv.y * f32(shape.high)), shape.high - 1u);
  // Collapse the limbs for display: the eye needs one number, and this
  // is the one place rounding the expansion costs nothing.
  var phase = 0.0;
  for (var k: u32 = 0u; k < shape.limbs; k = k + 1u) {
    phase = phase + feeds[offsets[k] + y * shape.wide + x];
  }
  let tau = 6.283185307179586;
  let turn = phase / tau - floor(phase / tau);
  let r = 0.5 + 0.5 * cos(tau * turn);
  let g = 0.5 + 0.5 * cos(tau * (turn - 0.3333333));
  let b = 0.5 + 0.5 * cos(tau * (turn - 0.6666667));
  return vec4<f32>(r, g, b, 1.0);
}
"""


def page(kernel_wgsl: str, plan: dict, width: int, height: int,
         limbs: int, digits: int, terms: int, seed: int,
         spread: float, published: int) -> str:
    manifest = {
        "wide": width, "high": height, "limbs": limbs, "digits": digits,
        "terms": terms, "seed": seed, "spread": spread,
        "slots": plan["slots"], "published": published,
    }
    return _PAGE.replace("__MANIFEST__", json.dumps(manifest)) \
                .replace("__KERNEL_WGSL__", json.dumps(kernel_wgsl)) \
                .replace("__GATHER_WGSL__", json.dumps(GATHER_WGSL)) \
                .replace("__PRESENT_WGSL__", json.dumps(PRESENT_WGSL))


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kuramoto field, compiled to WebGPU</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0b0d12; color: #d7dbe4;
         font: 14px/1.5 ui-sans-serif, system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center;
         gap: 12px; padding: 20px; }
  canvas { width: min(80vmin, 640px); height: min(80vmin, 640px);
           image-rendering: pixelated; border-radius: 6px;
           box-shadow: 0 0 0 1px #232838; }
  #status { font-variant-numeric: tabular-nums; min-height: 1.5em; }
  .note { max-width: 640px; color: #8d95a8; font-size: 13px; }
  b { color: #e7ebf4; font-weight: 600; }
</style>
</head>
<body>
<canvas id="view" width="512" height="512"></canvas>
<div id="status">starting…</div>
<div class="note">
  Every cell is pulled toward its four neighbours by
  <b>sin(their phase − mine)</b>. The sine is derived in SymPy, lowered to
  AbstractTensor Python, and compiled to WGSL by the repository's own
  WebGPU backend — at <b>Precision[2, float32]</b>, two 24-bit limbs,
  because WGSL has no f64 and two limbs carry more significand than the
  f64 they replace. The split two_product means no fused multiply-add
  appears, so WGSL's permission to double-round <code>fma()</code> never
  applies.
</div>
<script type="module">
const MANIFEST = __MANIFEST__;
const KERNEL_WGSL = __KERNEL_WGSL__;
const GATHER_WGSL = __GATHER_WGSL__;
const PRESENT_WGSL = __PRESENT_WGSL__;

const status = document.getElementById("status");
const canvas = document.getElementById("view");

function fail(message) {
  status.textContent = message;
  status.style.color = "#ff9d9d";
  throw new Error(message);
}

// A deterministic generator, so a run is reproducible and the page does
// not depend on the host's RNG.
function splitmix32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x9e3779b9) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function main() {
  if (!navigator.gpu) fail("WebGPU is unavailable in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fail("no WebGPU adapter");

  // The emitted ABI binds one storage buffer per formal, and a wide
  // kernel has many, so the device is asked for what the adapter can
  // actually give rather than the conservative default.
  const wanted = {};
  for (const key of ["maxStorageBuffersPerShaderStage",
                     "maxBindGroupsPlusVertexBuffers",
                     "maxBufferSize", "maxStorageBufferBindingSize"]) {
    if (adapter.limits[key] !== undefined) wanted[key] = adapter.limits[key];
  }
  const device = await adapter.requestDevice({ requiredLimits: wanted });
  device.addEventListener?.("uncapturederror", (e) =>
    fail("WebGPU error: " + e.error.message));

  const { wide, high, limbs } = MANIFEST;
  const count = wide * high;

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const storage = (bytes) => device.createBuffer({
    size: Math.max(bytes, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
         | GPUBufferUsage.COPY_SRC,
  });

  // Initial condition: phases uniform on a full turn, natural
  // frequencies normal. Both generated here so the field is the same
  // field every run.
  const random = splitmix32(MANIFEST.seed);
  const gaussian = () => {
    const u = Math.max(random(), 1e-12), v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const phases = new Float32Array(count);
  const spins = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    phases[i] = (random() * 2 - 1) * Math.PI;
    spins[i] = gaussian() * MANIFEST.spread;
  }

  // The feeds are ONE SPAN: formal at position k occupies
  // [k * count, (k + 1) * count). Constants are written once here and
  // never touched again; only theta's limbs are rewritten per step.
  const slots = MANIFEST.slots;
  const feeds = storage(slots.length * count * 4);
  const staging = new Float32Array(slots.length * count);
  const thetaSlots = [];
  for (const slot of slots) {
    const base = slot.at * count;
    if (slot.field === undefined) {
      staging.fill(slot.value, base, base + count);
    } else if (slot.field === "theta") {
      if (slot.limb === 0) staging.set(phases, base);
      thetaSlots[slot.limb] = base;
    } else if (slot.field === "omega") {
      if (slot.limb === 0) staging.set(spins, base);
    }
    // Neighbour fields are filled by the gather pass each step.
  }
  device.queue.writeBuffer(feeds, 0, staging);

  // Where the gather pass writes, and where it reads the phase from.
  const neighbourSlots = {};
  for (const slot of slots) {
    if (slot.field && slot.field !== "theta" && slot.field !== "omega") {
      neighbourSlots[slot.field] = neighbourSlots[slot.field] || [];
      neighbourSlots[slot.field][slot.limb] = slot.at * count;
    }
  }

  // The kernel publishes its two limbs to two output buffers; they are
  // copied back into theta's slots between steps, which is also the
  // ping-pong that stops a cell reading a neighbour it just overwrote.
  // ONE buffer per value the shader actually publishes. Binding more
  // than the layout declares is rejected outright, and the compute pass
  // is then dropped every frame while the step counter still advances --
  // which reads exactly like a field that refuses to organise.
  const outputs = [];
  for (let i = 0; i < MANIFEST.published; i++) outputs.push(storage(count * 4));

  const shape = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shape, 0,
    new Uint32Array([wide, high, count, limbs]));

  // The real slot of every limb of every field, in the order the shaders
  // index it. Nothing here assumes a field's limbs are adjacent.
  const order = [];
  for (const name of ["theta", "up", "down", "left", "right"]) {
    const row = name === "theta" ? thetaSlots : neighbourSlots[name];
    for (let limb = 0; limb < limbs; limb++) order.push(row[limb]);
  }
  const offsets = device.createBuffer({
    size: Math.max(order.length * 4, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(offsets, 0, new Uint32Array(order));

  status.textContent = "compiling shaders…";
  const kernelModule = device.createShaderModule({ code: KERNEL_WGSL });
  const info = await kernelModule.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    fail("kernel WGSL: " + errors[0].message + " (line " +
         errors[0].lineNum + ")");
  }
  const gatherModule = device.createShaderModule({ code: GATHER_WGSL });
  const presentModule = device.createShaderModule({ code: PRESENT_WGSL });

  const kernelPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: kernelModule, entryPoint: "main" } });
  const gatherPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: gatherModule, entryPoint: "main" } });
  const presentPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: presentModule, entryPoint: "vs" },
    fragment: { module: presentModule, entryPoint: "fs",
                targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // One feed span in, one buffer per published limb out.
  const kernelEntries = [{ binding: 0, resource: { buffer: feeds } }];
  outputs.forEach((buffer, index) =>
    kernelEntries.push({ binding: 1 + index, resource: { buffer } }));

  // A readback hook, so the field can be SCORED rather than watched:
  // it collapses theta's limbs on the host and reports the same local
  // coherence the CPU demos print, which is what makes "the GPU agrees"
  // a measurement instead of an impression.
  const readback = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  window.__field = async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      feeds, thetaSlots[0] * 4, readback, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    const at = (r, c) => values[((r + high) % high) * wide + ((c + wide) % wide)];
    let total = 0;
    for (let r = 0; r < high; r++) for (let c = 0; c < wide; c++) {
      const p = at(r, c);
      total += Math.cos(at(r - 1, c) - p) + Math.cos(at(r + 1, c) - p)
             + Math.cos(at(r, c - 1) - p) + Math.cos(at(r, c + 1) - p);
    }
    let finite = 0, nan = 0, lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (Number.isFinite(v)) { finite++; if (v < lo) lo = v; if (v > hi) hi = v; }
      else nan++;
    }
    return { step, coherence: total / (4 * count),
             finite, nan, lo, hi, first: Array.from(values.slice(0, 6)) };
  };

  let step = 0;
  let last = performance.now();
  let frames = 0;

  function frame() {
    const encoder = device.createCommandEncoder();

    // 1. neighbours off the torus (data movement)
    const gather = encoder.beginComputePass();
    gather.setPipeline(gatherPipeline);
    gather.setBindGroup(0, device.createBindGroup({
      layout: gatherPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shape } },
        { binding: 1, resource: { buffer: feeds } },
        { binding: 2, resource: { buffer: offsets } },
      ],
    }));
    gather.dispatchWorkgroups(Math.ceil(count / 64));
    gather.end();

    // 2. the compiled advance (the mathematics)
    const advance = encoder.beginComputePass();
    advance.setPipeline(kernelPipeline);
    advance.setBindGroup(0, device.createBindGroup({
      layout: kernelPipeline.getBindGroupLayout(0),
      entries: kernelEntries,
    }));
    advance.dispatchWorkgroups(Math.ceil(count / 256));
    advance.end();

    // 3. the advanced phase becomes the field, limb by limb
    for (let limb = 0; limb < outputs.length; limb++) {
      encoder.copyBufferToBuffer(
        outputs[limb], 0, feeds, thetaSlots[limb] * 4, count * 4);
    }

    // 4. paint it
    const paint = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    paint.setPipeline(presentPipeline);
    paint.setBindGroup(0, device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shape } },
        { binding: 1, resource: { buffer: feeds } },
        { binding: 2, resource: { buffer: offsets } },
      ],
    }));
    paint.draw(6);
    paint.end();

    device.queue.submit([encoder.finish()]);

    step += 1;
    frames += 1;
    const now = performance.now();
    if (now - last > 500) {
      const fps = frames * 1000 / (now - last);
      const sines = 4 * count * fps;
      status.textContent =
        `${wide}x${high} = ${count.toLocaleString()} cells · step ` +
        `${step.toLocaleString()} · ${fps.toFixed(0)} fps · ` +
        `${(sines / 1e6).toFixed(1)}M sines/s · ` +
        `${MANIFEST.limbs} f32 limbs, ${MANIFEST.terms} terms`;
      last = now; frames = 0;
    }
    schedule();
  }
  // rAF is SUSPENDED in a hidden tab, so a page driven only by it never
  // starts when nothing is looking at it -- the field would sit forever
  // showing whatever the last status write said. setTimeout keeps running
  // either way, so visibility decides the pacing and never the progress.
  window.__paused = false;
  function schedule() {
    if (window.__paused) { setTimeout(schedule, 50); return; }
    if (document.hidden) setTimeout(frame, 16);
    else requestAnimationFrame(frame);
  }
  schedule();
}

main().catch((error) => fail(String(error && error.message || error)));
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--limbs", type=int, default=2)
    parser.add_argument("--digits", type=int, default=14)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path,
                        default=Path("build/kuramoto-webgpu"))
    arguments = parser.parse_args(argv)

    width, height = int(arguments.width), int(arguments.height)
    cells = width * height
    limbs = max(2, int(arguments.limbs))

    print(f"emitting {width}x{height} = {cells:,d} cells at "
          f"{limbs} float32 limbs", flush=True)

    (emitted, flat, entry, sine, cosine, constants, terms,
     narrowed) = build(limbs, int(arguments.digits), cells)

    print(f"narrowed {narrowed} values to float32; "
          f"{terms} terms per series", flush=True)
    if not emitted.complete:
        for shortfall in emitted.shortfalls[:6]:
            print(f"  SHORTFALL {shortfall.operation}: "
                  f"{shortfall.reason[:80]}", flush=True)
        raise SystemExit("WGSL emission incomplete")

    quarter_exact = constant_rational("tau", int(arguments.digits)) / 4
    plan, ids = feed_plan(
        flat[entry], entry, limbs, sine, cosine, constants, quarter_exact,
        float(arguments.coupling), float(arguments.dt),
    )
    if plan["unbound"]:
        raise SystemExit(f"unbound formals: {sorted(plan['unbound'])[:8]}")

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kuramoto.wgsl").write_text(
        emitted.source, encoding="utf-8", newline="\n")
    html = page(emitted.source, plan, width, height, limbs,
                int(arguments.digits), terms, int(arguments.seed),
                float(arguments.spread), len(_returned_values(flat[entry])))
    path = destination / "index.html"
    path.write_text(html, encoding="utf-8", newline="\n")

    fields = sum(1 for slot in plan["slots"] if "field" in slot)
    print(f"shader {len(emitted.source):,d} bytes, "
          f"{len(plan['slots'])} feed slots ({fields} field, "
          f"{len(plan['slots']) - fields} constant) in one span",
          flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
