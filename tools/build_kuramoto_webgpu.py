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
    core_terms, hamiltonian_program, kicked_rotor_energy, kuramoto_program,
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

#: The gather shader, generated from a program's stencil. Nothing about
#: the neighbourhood is written into the shader by hand: a program that
#: wants diagonals, or a wider ring, or a second species, states it in its
#: stencil and this follows.
def gather_wgsl(stencil) -> str:
    reads = []
    for position, (name, (drow, dcol)) in enumerate(stencil.items(), start=1):
        row = ("row" if drow == 0 else
               f"(row + shape.high - {abs(drow)}u) % shape.high" if drow < 0
               else f"(row + {drow}u) % shape.high")
        col = ("col" if dcol == 0 else
               f"(col + shape.wide - {abs(dcol)}u) % shape.wide" if dcol < 0
               else f"(col + {dcol}u) % shape.wide")
        reads.append(
            f"    // {name} at ({drow:+d}, {dcol:+d})\n"
            f"    feeds[offsets[{position}u * shape.limbs + k] + i] = "
            f"feeds[src + ({row}) * shape.wide + ({col})];"
        )
    return """
// Offsets into the one feed span, supplied explicitly. A field's limbs
// are NOT consecutive -- the emitter lays out limb 0 of every formal
// before limb 1 of any of them -- so striding by the limb index walks
// into a NEIGHBOUR's slot and then into the coefficients. Measured, that
// overwrote the series with phase data every frame and the field came
// back entirely NaN.
struct Shape { wide: u32, high: u32, count: u32, limbs: u32 };
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> feeds: array<f32>;
// [source..., then one block per stencil entry], `limbs` entries each.
@group(0) @binding(2) var<storage, read> offsets: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= shape.count) { return; }
  let row = i / shape.wide;
  let col = i % shape.wide;
  // Every limb moves together: a gather is data movement and cannot
  // round, so the expansion crosses it intact.
  for (var k: u32 = 0u; k < shape.limbs; k = k + 1u) {
    let src = offsets[k];
""" + "\n".join(reads) + """
  }
}
"""


def cell_source(equation, width: int, element: str) -> tuple[str, tuple]:
    """The per-cell advance as an annotated AbstractTensor function.

    No loop: a compute invocation IS the loop body, and the emitted entry
    already derives ``linear_index`` from the invocation id.
    """

    equations = (list(equation) if isinstance(equation, (list, tuple))
                 else [equation])
    compiled = compile_sympy_equations(equations, name="kuramoto_step")
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


#: The programs this page can carry. Each is one FieldProgram, and
#: nothing below here knows which was chosen.
def _program_for(choice: str, terms: int, lag: bool):
    if choice == "rotor":
        # A Hamiltonian, not an update: SymPy derives the whole step.
        return hamiltonian_program(
            kicked_rotor_energy(), terms, scalars=("K", "epsilon"),
            name="kicked-rotor",
        )
    return kuramoto_program(terms, lag=lag)


def build(width: int, digits: int, cells: int, lag: bool = False,
          choice: str = "kuramoto", local_size: int = 256):
    """Lower, narrow, flatten, and emit WGSL for one FieldProgram."""

    sine = list(core_terms("sin", digits))
    cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))

    program = _program_for(choice, terms, lag)
    equation, constants = program.equation, program.constants

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
        outputs={entry: returned}, preferred_local_size=local_size,
    )
    return (emitted, flat, entry, sine, cosine, constants, terms, narrowed,
            program)


def writeback_plan(function, program, limbs: int):
    """Which field and limb each published value belongs to.

    The Ret names one value per equation, in equation order, and each
    expands to its limb row -- so the published order is field-major and
    the host can be told it rather than having to infer it.
    """

    rows = dict(function.metadata.get("precision_lowered_values") or ())
    plan = []
    for block in function.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) not in ("Ret", "Return"):
                continue
            for field, value in zip(program.advances, instruction.args):
                row = rows.get(int(value.id)) or (int(value.id),)
                for limb, _identifier in enumerate(row):
                    plan.append({"field": field, "limb": limb})
            return plan
    return plan


def _returned_values(function):
    """Everything the step publishes -- EVERY LIMB, not the collapsed head.

    Precision lowering ends each wide operation by collapsing its limbs
    into one value so it can be used as an ordinary scalar, and the
    ``Ret`` names that collapsed value. Publishing it alone throws the
    expansion away at the boundary: the arithmetic inside a step is as
    wide as it was asked to be, and the field between steps is a single
    float. A ladder that cannot carry its own state forward buys nothing
    but a slower first step.

    The limb row is recorded per value by the lowering, so the row is
    what gets published when there is one.
    """

    rows = dict(function.metadata.get("precision_lowered_values") or ())
    produced = {
        int(instruction.res.id): instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    for block in function.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) not in ("Ret", "Return"):
                continue
            published = []
            for value in instruction.args:
                row = rows.get(int(value.id))
                if not row:
                    published.append(value)
                    continue
                published.extend(
                    produced.get(int(limb), value) for limb in row
                )
            return published
    return []


def feed_plan(function, entry: str, width: int, sine, cosine, constants,
              quarter_exact, runtime: dict, program):
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
        name: limbs_of(Fraction(value).limit_denominator(10 ** 9))
        for name, value in runtime.items()
    }
    scalars.update({
        "quarter": limbs_of(quarter_exact),
        "neg_quarter": limbs_of(-quarter_exact),
        "inv_quarter": limbs_of(1 / quarter_exact),
    })
    for prefix, values in (("c", sine), ("d", cosine)):
        for index, value in enumerate(values):
            scalars[f"{prefix}{index}"] = limbs_of(value)
    for name, value in constants.items():
        scalars[name] = limbs_of(value)

    # The feeds are ONE SPAN: formal at position k occupies
    # [k * count, (k + 1) * count). So the plan is written by POSITION in
    # the emitted signature, not by SSA id, and the host fills one buffer.
    fields = set(program.state) | set(program.stencil)
    by_id = {int(identifier): name for name, identifier in ids.items()}
    plan = {"slots": [], "unbound": []}
    for position, formal in enumerate(function.args):
        name = by_id.get(int(formal.id))
        if name is None:
            plan["unbound"].append(f"%{formal.id}")
            continue
        base, _, suffix = name.partition("__limb")
        limb = int(suffix) if suffix else 0
        if base in fields:
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


TILE_WGSL = """
struct View {
  width: f32,
  height: f32,
  tile: f32,
  _padding: f32,
};
@group(0) @binding(0) var field_sampler: sampler;
@group(0) @binding(1) var field_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> view: View;

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

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  let repeats = vec2<f32>(view.width, view.height) / view.tile;
  return textureSample(field_texture, field_sampler, in.uv * repeats);
}
"""


PERTURB_WGSL = """
struct Shape { wide: u32, high: u32, count: u32, limbs: u32 };
struct Interaction {
  x: f32,
  y: f32,
  radius: f32,
  strength: f32,
  enabled: u32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
};
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> feeds: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<uniform> interaction: Interaction;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= shape.count || interaction.enabled == 0u) { return; }

  let row = f32(i / shape.wide);
  let col = f32(i % shape.wide);
  let raw_x = abs(col - interaction.x);
  let raw_y = abs(row - interaction.y);
  let dx = min(raw_x, f32(shape.wide) - raw_x);
  let dy = min(raw_y, f32(shape.high) - raw_y);
  let distance_2 = dx * dx + dy * dy;
  let radius_2 = interaction.radius * interaction.radius;
  if (distance_2 >= radius_2) { return; }

  let falloff = 1.0 - distance_2 / radius_2;
  let delta = interaction.strength * falloff * falloff;
  let high_at = offsets[0] + i;
  let high_value = feeds[high_at];

  // Error-free two-sum keeps an external bump in the same two-limb phase
  // representation as the compiled program. One-limb builds simply use it.
  if (shape.limbs > 1u) {
    let low_at = offsets[1] + i;
    let first = high_value + delta;
    let virtual_delta = first - high_value;
    let error = (high_value - (first - virtual_delta)) +
                (delta - virtual_delta);
    let corrected_low = feeds[low_at] + error;
    let result = first + corrected_low;
    feeds[high_at] = result;
    feeds[low_at] = corrected_low - (result - first);
  } else {
    feeds[high_at] = high_value + delta;
  }
}
"""


def page(kernel_wgsl: str, plan: dict, width: int, height: int,
         limbs: int, digits: int, terms: int, seed: int,
         spread: float, published: int, program, writeback,
         workgroup_size: int) -> str:
    # A stamp over the shader AND the host, so a page can say which build
    # it is. Every confusing round here has had a stale artifact as a live
    # hypothesis, and a page that cannot identify itself keeps that
    # hypothesis alive for free.
    import hashlib

    build_stamp = hashlib.sha256(
        (kernel_wgsl + _PAGE + json.dumps(plan, sort_keys=True))
        .encode("utf-8")
    ).hexdigest()[:8]
    manifest = {
        "wide": width, "high": height, "limbs": limbs, "digits": digits,
        "workgroupSize": workgroup_size,
        "terms": terms, "seed": seed, "spread": spread,
        "slots": plan["slots"], "published": published,
        "build": build_stamp,
        "advances": list(program.advances),
        "gatherFrom": program.advances[0],
        "stateSeed": {
            field: ("turn" if field == program.advances[0]
                    else "spread" if field == "omega" else "quiet")
            for field in program.state
        },
        "writeback": writeback,
        "stencil": list(program.stencil),
        "programName": program.name,
    }
    page_title = {
        "kuramoto": "Kuramoto phase field",
        "sakaguchi": "Sakaguchi-Kuramoto phase field",
        "kicked-rotor": "Hamiltonian kicked-rotor lattice",
    }[program.name]
    program_note = {
        "kuramoto": (
            "Every cell is pulled toward its four neighbours by "
            "<b>sin(their phase - mine)</b>."
        ),
        "sakaguchi": (
            "Every cell is pulled toward its four neighbours through a "
            "<b>phase-lagged sine coupling</b>, breaking the ordinary "
            "Kuramoto symmetry."
        ),
        "kicked-rotor": (
            "Each cell is a <b>Hamiltonian kicked rotor</b>: the compiled "
            "step advances both its angle and conjugate momentum while "
            "coupling it to the surrounding lattice."
        ),
    }[program.name]
    return _PAGE.replace("__PAGE_TITLE__", page_title) \
                .replace("__PROGRAM_NOTE__", program_note) \
                .replace("__MANIFEST__", json.dumps(manifest)) \
                .replace("__KERNEL_WGSL__", json.dumps(kernel_wgsl)) \
                .replace("__GATHER_WGSL__",
                         json.dumps(gather_wgsl(program.stencil))) \
                .replace("__PRESENT_WGSL__", json.dumps(PRESENT_WGSL)) \
                .replace("__TILE_WGSL__", json.dumps(TILE_WGSL)) \
                .replace("__PERTURB_WGSL__", json.dumps(PERTURB_WGSL))


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__PAGE_TITLE__, compiled to WebGPU</title>
<style>
  :root { color-scheme: dark; }
  html, body { width: 100%; min-height: 100%; }
  body { margin: 0; overflow: hidden; background: #0b0d12; color: #d7dbe4;
         font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }
  canvas { position: fixed; inset: 0; width: 100vw; height: 100vh;
           image-rendering: pixelated; touch-action: none; cursor: crosshair; }
  main { position: relative; z-index: 1; width: min(620px, calc(100vw - 40px));
         margin: 20px auto; padding: 14px 16px; border-radius: 8px;
         background: rgb(11 13 18 / .78); border: 1px solid rgb(215 219 228 / .16);
         box-shadow: 0 12px 48px rgb(0 0 0 / .28); backdrop-filter: blur(9px);
         pointer-events: none; }
  #status, #health { font-variant-numeric: tabular-nums;
                     min-height: 1.5em; }
  #health { color: #8d95a8; }
  #stamp { color: #5d6478; font-size: 12px; }
  #pointer { color: #c9d2e7; font-variant-numeric: tabular-nums; }
  .note { max-width: 640px; color: #a6aec0; font-size: 13px; }
  b { color: #e7ebf4; font-weight: 600; }
</style>
</head>
<body>
<canvas id="view" width="512" height="512"></canvas>
<main>
<div id="status">loading…</div>
<div id="health"></div>
<div id="pointer">move over any tile for a hill · press for a dimple</div>
<div id="stamp">build …</div>
<div class="note">
  __PROGRAM_NOTE__ The update is derived in SymPy, lowered to
  AbstractTensor Python, and compiled to WGSL by the repository's own
  WebGPU backend — at <b>Precision[2, float32]</b>, two 24-bit limbs,
  because WGSL has no f64 and two limbs carry more significand than the
  f64 they replace. The split two_product means no fused multiply-add
  appears, so WGSL's permission to double-round <code>fma()</code> never
  applies.
</div>
</main>
<script type="module">
const MANIFEST = __MANIFEST__;
const KERNEL_WGSL = __KERNEL_WGSL__;
const GATHER_WGSL = __GATHER_WGSL__;
const PRESENT_WGSL = __PRESENT_WGSL__;
const TILE_WGSL = __TILE_WGSL__;
const PERTURB_WGSL = __PERTURB_WGSL__;

const status = document.getElementById("status");
const health = document.getElementById("health");
const pointerStatus = document.getElementById("pointer");
const stamp = document.getElementById("stamp");
stamp.textContent = "build " + MANIFEST.build;
const canvas = document.getElementById("view");
let failed = false;

function showFailure(message) {
  failed = true;
  status.textContent = message;
  status.style.color = "#ff9d9d";
  console.error(message);
}

function fail(message) {
  showFailure(message);
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

  // Ask for NOTHING beyond the defaults. Requesting every limit the
  // adapter reports was needed when the ABI bound one storage buffer per
  // formal and this kernel wanted sixty-nine of them. The feeds are one
  // packed span now, so five bindings is the whole requirement -- well
  // inside what WebGPU guarantees -- and a maximal request is just
  // pressure on a device several pages may be sharing.
  const device = await adapter.requestDevice();
  window.__gpu = { adapter, device };
  device.lost.then(info => showFailure(
    "WebGPU device lost: " + (info.message || info.reason || "unknown")));
  device.addEventListener?.("uncapturederror", (e) =>
    showFailure("WebGPU error: " + e.error.message));
  // Nothing here may fail quietly. A rejection with no handler used to
  // stop the loop and leave a cleared canvas looking like a finished
  // render.
  window.addEventListener("unhandledrejection", (e) =>
    showFailure("unhandled: " + String(e.reason && e.reason.message || e.reason)));

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
  // One seed array per state field, named by how the program wants it
  // started: a phase is uniform on a turn, anything else is drawn narrow.
  const seeds = {};
  for (const [field, kind] of Object.entries(MANIFEST.stateSeed)) {
    const values = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      values[i] = kind === "turn" ? (random() * 2 - 1) * Math.PI
                : kind === "spread" ? gaussian() * MANIFEST.spread
                : 0;
    }
    seeds[field] = values;
  }

  // The feeds are ONE SPAN: formal at position k occupies
  // [k * count, (k + 1) * count). Constants are written once here and
  // never touched again; only theta's limbs are rewritten per step.
  const slots = MANIFEST.slots;
  const feeds = storage(slots.length * count * 4);
  const staging = new Float32Array(slots.length * count);
  const stateSlots = {};
  for (const slot of slots) {
    const base = slot.at * count;
    if (slot.field === undefined) {
      staging.fill(slot.value, base, base + count);
    } else if (MANIFEST.stateSeed[slot.field] !== undefined) {
      if (slot.limb === 0) staging.set(seeds[slot.field], base);
      stateSlots[slot.field] = stateSlots[slot.field] || [];
      stateSlots[slot.field][slot.limb] = base;
    }
    // Neighbour fields are filled by the gather pass each step.
  }
  device.queue.writeBuffer(feeds, 0, staging);

  // Where the gather pass writes, and where it reads the phase from.
  const neighbourSlots = {};
  for (const slot of slots) {
    if (slot.field && MANIFEST.stencil.includes(slot.field)) {
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
  // Source field first, then one block per stencil entry, in the order
  // the generated gather indexes them.
  for (const name of [MANIFEST.gatherFrom, ...MANIFEST.stencil]) {
    const row = stateSlots[name] || neighbourSlots[name];
    for (let limb = 0; limb < limbs; limb++) order.push(row[limb]);
  }
  const offsets = device.createBuffer({
    size: Math.max(order.length * 4, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(offsets, 0, new Uint32Array(order));

  // "loading", not "compiling": the page does not compile anything.
  // It hands finished WGSL to the browser, and this string is only a
  // default that stands until the first frame overwrites it -- so it
  // must not claim to know what is happening.
  status.textContent = "loading…";
  const kernelModule = device.createShaderModule({ code: KERNEL_WGSL });
  const info = await kernelModule.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    fail("kernel WGSL: " + errors[0].message + " (line " +
         errors[0].lineNum + ")");
  }
  const gatherModule = device.createShaderModule({ code: GATHER_WGSL });
  const presentModule = device.createShaderModule({ code: PRESENT_WGSL });
  const tileModule = device.createShaderModule({ code: TILE_WGSL });
  const perturbModule = device.createShaderModule({ code: PERTURB_WGSL });

  async function requireValidShader(label, module) {
    const report = await module.getCompilationInfo();
    const errors = report.messages.filter(message => message.type === "error");
    if (!errors.length) return;
    const first = errors[0];
    fail(`${label} WGSL: ${first.message} (line ${first.lineNum})`);
  }
  await requireValidShader("gather", gatherModule);
  await requireValidShader("present", presentModule);
  await requireValidShader("tile", tileModule);
  await requireValidShader("perturb", perturbModule);

  const kernelPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: kernelModule, entryPoint: "main" } });
  const gatherPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: gatherModule, entryPoint: "main" } });
  const perturbPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: perturbModule, entryPoint: "main" } });
  const presentPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: presentModule, entryPoint: "vs" },
    fragment: { module: presentModule, entryPoint: "fs",
                targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const tilePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: tileModule, entryPoint: "vs" },
    fragment: { module: tileModule, entryPoint: "fs",
                targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // The simulation publishes one logical image at its own draw rate. That
  // image stays on the GPU and is sampled repeatedly across the page in the
  // same command buffer, so every tile is the exact same system step.
  const fieldTexture = device.createTexture({
    size: [wide, high], format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const fieldView = fieldTexture.createView();
  const fieldSampler = device.createSampler({
    addressModeU: "repeat", addressModeV: "repeat",
    magFilter: "nearest", minFilter: "nearest",
  });
  const viewUniform = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const interactionUniform = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const tileCssPixels = 256;
  const pointer = { inside: false, pressed: false, x: 0, y: 0 };

  function reportPointer(event, forceInside = null) {
    const rect = canvas.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const tileX = ((localX % tileCssPixels) + tileCssPixels) % tileCssPixels;
    const tileY = ((localY % tileCssPixels) + tileCssPixels) % tileCssPixels;
    pointer.x = tileX / tileCssPixels * wide;
    pointer.y = tileY / tileCssPixels * high;
    pointer.inside = forceInside ?? (localX >= 0 && localY >= 0 &&
      localX < rect.width && localY < rect.height);
    window.__pointer = {
      page: { x: localX, y: localY },
      field: { x: pointer.x, y: pointer.y },
      pressed: pointer.pressed,
    };
    window.dispatchEvent(new CustomEvent("turing-pointer", {
      detail: window.__pointer,
    }));
    pointerStatus.textContent =
      `page ${localX.toFixed(0)}, ${localY.toFixed(0)} → field ` +
      `${pointer.x.toFixed(1)}, ${pointer.y.toFixed(1)} · ` +
      (pointer.pressed ? "dimple" : "hill");
  }
  canvas.addEventListener("pointermove", reportPointer);
  canvas.addEventListener("pointerenter", reportPointer);
  canvas.addEventListener("pointerdown", event => {
    pointer.pressed = true;
    canvas.setPointerCapture(event.pointerId);
    reportPointer(event);
  });
  const releasePointer = event => {
    pointer.pressed = false;
    reportPointer(event);
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);
  canvas.addEventListener("pointerleave", event => {
    reportPointer(event, false);
  });

  // One feed span in, one buffer per published limb out.
  const kernelEntries = [{ binding: 0, resource: { buffer: feeds } }];
  outputs.forEach((buffer, index) =>
    kernelEntries.push({ binding: 1 + index, resource: { buffer } }));

  // A readback hook, so the field can be SCORED rather than watched:
  // it collapses theta's limbs on the host and reports the same local
  // coherence the CPU demos print, which is what makes "the GPU agrees"
  // a measurement instead of an impression.
  // A fresh staging buffer per call, mapped once and destroyed. Sharing
  // one buffer across overlapping calls is what broke this: the status
  // line asks every half second, a call takes longer than that, and the
  // second map lands on a buffer the first still holds. The probe was
  // reporting a fault in itself as a fault in the field.
  let probing = false;
  async function readField(slot) {
    const staging = device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(feeds, slot * 4, staging, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return values;
  }

  window.__field = async () => {
    if (probing) return null;
    probing = true;
    try {
      const slots = stateSlots[MANIFEST.gatherFrom];
      const values = await readField(slots[0]);
      const low = limbs > 1 ? await readField(slots[1]) : new Float32Array(count);

      const at = (r, c) =>
        values[((r + high) % high) * wide + ((c + wide) % wide)];
      let total = 0, finite = 0, nan = 0, lo = Infinity, hi = -Infinity;
      for (let r = 0; r < high; r++) for (let c = 0; c < wide; c++) {
        const v = at(r, c);
        total += Math.cos(at(r - 1, c) - v) + Math.cos(at(r + 1, c) - v)
               + Math.cos(at(r, c - 1) - v) + Math.cos(at(r, c + 1) - v);
      }
      for (const v of values) {
        if (Number.isFinite(v)) { finite++; if (v < lo) lo = v; if (v > hi) hi = v; }
        else nan++;
      }
      let lowNonzero = 0;
      for (const v of low) if (v !== 0) lowNonzero++;
      return { step, coherence: total / (4 * count), finite, nan, lo, hi,
               lowNonzero };
    } finally {
      probing = false;
    }
  };

  let step = 0;
  let last = performance.now();
  let frames = 0;

  // Deliberately NOT async. Awaiting completion here was added on a bad
  // measurement -- a hidden pane reporting 0 fps and a lost device, which
  // was the pane being throttled and reclaimed, not the shader. Worse, an
  // async frame turns any throw into an unhandled rejection: the loop
  // stops, nothing is reported, and the last cleared frame stays on
  // screen. A silent stop is the one failure this page must not have.
  function frame() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const displayWidth = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const displayHeight = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    device.queue.writeBuffer(viewUniform, 0, new Float32Array([
      canvas.clientWidth, canvas.clientHeight, tileCssPixels, 0,
    ]));
    const interaction = new ArrayBuffer(32);
    const interactionFloats = new Float32Array(interaction);
    const interactionWords = new Uint32Array(interaction);
    interactionFloats.set([
      pointer.x, pointer.y, 5.5, pointer.pressed ? -0.035 : 0.035,
    ]);
    interactionWords[4] = pointer.inside ? 1 : 0;
    device.queue.writeBuffer(interactionUniform, 0, interaction);

    const encoder = device.createCommandEncoder();

    // 1. A shell-owned intervention changes the real field. Page coordinates
    // are already folded back through the repeated texture period.
    const perturb = encoder.beginComputePass();
    perturb.setPipeline(perturbPipeline);
    perturb.setBindGroup(0, device.createBindGroup({
      layout: perturbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shape } },
        { binding: 1, resource: { buffer: feeds } },
        { binding: 2, resource: { buffer: offsets } },
        { binding: 3, resource: { buffer: interactionUniform } },
      ],
    }));
    perturb.dispatchWorkgroups(Math.ceil(count / 64));
    perturb.end();

    // 2. neighbours off the torus (data movement)
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

    // 3. the compiled advance (the mathematics)
    const advance = encoder.beginComputePass();
    advance.setPipeline(kernelPipeline);
    advance.setBindGroup(0, device.createBindGroup({
      layout: kernelPipeline.getBindGroupLayout(0),
      entries: kernelEntries,
    }));
    advance.dispatchWorkgroups(Math.ceil(count / MANIFEST.workgroupSize));
    advance.end();

    // 4. the advanced phase becomes the field, limb by limb
    // Each published value goes to the field and limb the manifest
    // names. Two-equation programs advance two fields, and every limb of
    // each is published, so nothing collapses on the way back in.
    MANIFEST.writeback.forEach((where, index) => {
      encoder.copyBufferToBuffer(
        outputs[index], 0, feeds,
        stateSlots[where.field][where.limb] * 4, count * 4);
    });

    // 5. Pull one texture from current system state.
    const publish = encoder.beginRenderPass({
      colorAttachments: [{
        view: fieldView, loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    publish.setPipeline(presentPipeline);
    publish.setBindGroup(0, device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shape } },
        { binding: 1, resource: { buffer: feeds } },
        { binding: 2, resource: { buffer: offsets } },
      ],
    }));
    publish.draw(6);
    publish.end();

    // 6. Tile that exact texture across the page in the same submission.
    const paint = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    paint.setPipeline(tilePipeline);
    paint.setBindGroup(0, device.createBindGroup({
      layout: tilePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: fieldSampler },
        { binding: 1, resource: fieldView },
        { binding: 2, resource: { buffer: viewUniform } },
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
      // Say what the FIELD contains, not just that frames happened. A
      // page that reports progress while its numbers are NaN is how two
      // separate failures here looked exactly like success.
      if (MANIFEST.programName === "kicked-rotor") {
        // This kernel is large enough that a periodic CPU map can become the
        // pacing bottleneck. Its angle and momentum remain GPU-resident; the
        // texture is the observation path and explicit __field() remains
        // available when a diagnostic readback is actually requested.
        health.textContent =
          "field: GPU-resident angle + momentum · interactive phase forcing";
      } else {
        window.__field().then((f) => {
          if (!f) return;
          stamp.textContent = "build " + MANIFEST.build;
          health.textContent = f.nan
            ? `field: ${f.nan} of ${count} cells are NaN`
            : `field: [${f.lo.toFixed(2)}, ${f.hi.toFixed(2)}] · ` +
              `low limb live in ${f.lowNonzero}/${count} · ` +
              `coherence ${f.coherence.toFixed(4)}`;
        }).catch((e) => {
          health.textContent = "field: " + String(e).slice(0, 80);
        });
      }
      status.textContent =
        `${wide}x${high} = ${count.toLocaleString()} cells · step ` +
        `${step.toLocaleString()} · ${fps.toFixed(0)} fps · ` +
        `${(sines / 1e6).toFixed(1)}M sines/s · ` +
        `${MANIFEST.limbs} f32 limbs, ${MANIFEST.terms} terms`;
      last = now; frames = 0;
    }
    // A program step is a transaction: do not enqueue the next one until
    // every writeback and the corresponding observation frame has finished.
    // Small kernels can hide an unbounded submission backlog for a long time;
    // a larger compiled program makes the same harness bug lose the device.
    // Keep the promise explicit (rather than making frame async) so failures
    // are latched and reported instead of becoming a rejected animation task.
    device.queue.onSubmittedWorkDone().then(schedule).catch((error) =>
      showFailure("WebGPU submission failed: " +
        String(error && error.message || error)));
  }
  // rAF is SUSPENDED in a hidden tab, so a page driven only by it never
  // starts when nothing is looking at it -- the field would sit forever
  // showing whatever the last status write said. setTimeout keeps running
  // either way, so visibility decides the pacing and never the progress.
  window.__paused = false;
  function schedule() {
    if (failed) return;
    if (window.__paused) { setTimeout(schedule, 50); return; }
    if (document.hidden) setTimeout(frame, 16);
    else requestAnimationFrame(frame);
  }
  schedule();
}

main().catch((error) => showFailure(String(error && error.message || error)));
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--local-size", type=int, default=256,
                        help="preferred WebGPU compute workgroup width")
    parser.add_argument("--limbs", type=int, default=2)
    parser.add_argument("--digits", type=int, default=14)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--program", default="kuramoto",
                        choices=("kuramoto", "rotor"),
                        help="kuramoto: a phase field. rotor: a lattice of "
                             "kicked rotors whose update SymPy derives from "
                             "a Hamiltonian")
    parser.add_argument("--kick", type=float, default=1.2,
                        help="rotor only: the standard-map kick strength")
    parser.add_argument("--lag", type=float, default=0.0,
                        help="Sakaguchi phase lag in radians; any nonzero "
                             "value selects the lagged program")
    parser.add_argument("--output", type=Path,
                        default=Path("build/kuramoto-webgpu"))
    arguments = parser.parse_args(argv)

    width, height = int(arguments.width), int(arguments.height)
    cells = width * height
    limbs = max(2, int(arguments.limbs))

    print(f"emitting {width}x{height} = {cells:,d} cells at "
          f"{limbs} float32 limbs", flush=True)

    (emitted, flat, entry, sine, cosine, constants, terms, narrowed,
     program) = build(limbs, int(arguments.digits), cells,
                      lag=bool(arguments.lag),
                      choice=str(arguments.program),
                      local_size=int(arguments.local_size))

    print(f"narrowed {narrowed} values to float32; "
          f"{terms} terms per series", flush=True)
    if not emitted.complete:
        for shortfall in emitted.shortfalls[:6]:
            print(f"  SHORTFALL {shortfall.operation}: "
                  f"{shortfall.reason[:80]}", flush=True)
        raise SystemExit("WGSL emission incomplete")

    quarter_exact = constant_rational("tau", int(arguments.digits)) / 4
    runtime = {"dt": float(arguments.dt)}
    if "coupling" in program.scalars:
        runtime["coupling"] = float(arguments.coupling)
    if "alpha" in program.scalars:
        runtime["alpha"] = float(arguments.lag)
    if "K" in program.scalars:
        runtime["K"] = float(arguments.kick)
    if "epsilon" in program.scalars:
        runtime["epsilon"] = float(arguments.coupling)
    plan, ids = feed_plan(
        flat[entry], entry, limbs, sine, cosine, constants, quarter_exact,
        runtime, program,
    )
    if plan["unbound"]:
        raise SystemExit(f"unbound formals: {sorted(plan['unbound'])[:8]}")

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kuramoto.wgsl").write_text(
        emitted.source, encoding="utf-8", newline="\n")
    html = page(emitted.source, plan, width, height, limbs,
                int(arguments.digits), terms, int(arguments.seed),
                float(arguments.spread), len(_returned_values(flat[entry])),
                program, writeback_plan(flat[entry], program, limbs),
                int(emitted.launch_plan.workgroup_size[0]))
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
