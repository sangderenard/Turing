"""Build the repository's homepage: index.html.

    python build_homepage.py

One ordinary Python function is ingested as an AST, planned once as a
ProcessGraph, lowered through the AOT compiler, and emitted through every
backend this repository has. The page is what comes out: a WebAssembly
binary you can run in the browser, next to the Fortran, SPIR-V, GLSL, SSA
and NumPy/PyTorch/AbstractTensor source that came from the same compilation.

The kernel is a Mandelbrot escape count, chosen because it is short enough
to read in full and its output is a picture, so a wrong answer is visible
rather than a number nobody checks.

The page is a single self-contained file -- the ``.wasm`` is embedded as
base64 -- so GitHub Pages needs nothing but ``index.html`` at the repository
root, and it works opened from disk too.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
from pathlib import Path

import numpy as np

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.backend_sources import collect_backend_sources
from src.compiler.fused_program_wasm_backend import emit_wasm_module, required_steps
from src.compiler.shell_telemetry import TelemetryChannel, summarize_process_graph
from src.compiler.wasm_class_modules import (
    build_embedded_class_graph,
    describe_process_graph_api,
    emit_class_modules,
    partition_reduced_program,
)
from src.compiler.wasm_html_shell import emit_html_shell
from src.transmogrifier.graph.graph_express2 import ProcessGraph

# Every iteration is unrolled into the emitted program, so this is the one
# number that decides how large it gets. The escape test is the break-out:
# once a point has escaped, the comparison stops incrementing, so running
# more iterations refines the boundary rather than changing what it means.
ITERATIONS = 160
WASM_REGION_STEPS = 400
WASM_MODULE_DIR = "site-wasm"

# The orbit is clamped so a diverging point cannot reach infinity and poison
# the arithmetic; well above the escape radius, so it never touches a point
# that is still inside.
ORBIT_CLAMP = 1.0e18

KERNEL = f"""
def interest_network(unit_x, unit_y, interest):
    # Frozen network source enters the same AOT/WASM pipeline as the fractal.
    h0 = (unit_x * 0.83 + unit_y * -0.41 + interest * 0.72 + 0.15).tanh()
    h1 = (unit_x * -0.36 + unit_y * 0.91 + interest * 0.48 - 0.08).tanh()
    return (h0 * 0.62 + h1 * -0.57 + interest * 0.31).tanh()


def shade(count):
    # mandelbrot_jpeg_planes, ported: sqrt of the normalised escape count,
    # three cosine channels offset by 0, 0.21 and 0.43, scaled to 0-255 and
    # clamped, exactly as the original composed its display planes.
    #
    # The original raised each channel to 1.65. WebAssembly has no pow
    # instruction, and reaching it through exp/log would need log below the
    # quarter its table starts at -- which is where a colour ramp spends most
    # of its range. x^1.625 = x * sqrt(x) * sqrt(sqrt(sqrt(x))) uses only
    # sqrt and multiply, both native, and differs from x^1.65 by under two
    # percent across [0,1] -- below a quantisation step at 8 bits.
    phase = (count * {1.0 / ITERATIONS!r}).minimum(1.0).maximum(0.0).sqrt()

    wave_r = ((phase + 0.0) * 6.283185307179586).cos() * 0.5 + 0.5
    r_half = wave_r.sqrt()
    red = (wave_r * r_half * r_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    wave_g = ((phase + 0.21) * 6.283185307179586).cos() * 0.5 + 0.5
    g_half = wave_g.sqrt()
    green = (wave_g * g_half * g_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    wave_b = ((phase + 0.43) * 6.283185307179586).cos() * 0.5 + 0.5
    b_half = wave_b.sqrt()
    blue = (wave_b * b_half * b_half.sqrt().sqrt() * 255.0 + 0.5).minimum(255.0).maximum(0.0)

    # A tuple return does not lower, so one channel is returned and all three
    # remain as executed steps for the build to name as outputs.
    return red + green * 0.0 + blue * 0.0


def quadratic_family(unit_x, unit_y, center_x, center_y, span,
                     family_mix, julia_x, julia_y):
    # The continuous Mandelbrot-to-Julia quadratic family. family_mix = 0 is
    # the Mandelbrot set (orbit starts at zero, constant is the pixel); 1 is
    # a Julia set (orbit starts at the pixel, constant is fixed); everything
    # between is a real member of the family rather than a cross-fade of two
    # pictures.
    cx = center_x + unit_x * span
    cy = center_y + unit_y * span
    zx = cx * family_mix
    zy = cy * family_mix
    constant_x = cx + family_mix * (julia_x - cx)
    constant_y = cy + family_mix * (julia_y - cy)
    count = cx * 0.0
    clamp_value = cx * 0.0 + {ORBIT_CLAMP}
    for _ in range({ITERATIONS}):
        zx2 = zx * zx
        zy2 = zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        next_zx = zx2 - zy2 + constant_x
        next_zy = 2.0 * zx * zy + constant_y
        zx = next_zx.minimum(clamp_value).maximum(-clamp_value)
        zy = next_zy.minimum(clamp_value).maximum(-clamp_value)
    return count


def render(unit_x, unit_y, t, interest):
    # The page supplies the grid and the clock. Everything else -- where the
    # camera is, how deep, which member of the family -- is computed here.
    #
    # sin, cos and exp2 have no WebAssembly instruction; they arrive as
    # bounded lookup tables baked into this module, which is what lets the
    # whole trajectory live in the compiled program instead of being worked
    # out in JavaScript and fed in.
    #
    # Written as separate assignments rather than a tuple-returning helper:
    # simultaneous tuple assignment does not lower (see aot_compile).
    #
    # animated_camera + dream_parameters from demo_mandelbrot_fusion,
    # ported. The audio terms are the only omission -- the original modulated
    # three of these with bass/low_mid/high_mid, and reaction = 0 removes
    # exactly those. zoom_rate is 0 by default, so there is no progressive
    # dive: the span oscillates about the base and returns.
    #
    # t is TRAVEL, not a frame count; every frequency here is in travel units.
    log_zoom = (t * 0.71).sin() * 1.25 + (t * 1.93).sin() * 0.45
    mandelbrot_span = log_zoom.exp() * 0.004
    dx = ((t * 0.83).sin() * 0.58 + (t * 2.17).sin() * 0.22) * 0.004
    dy = ((t * 0.97 + 0.61).sin() * 0.48 + (t * 1.67).sin() * 0.19) * 0.004 - 0.0011
    mandelbrot_center_x = dx - 0.743643887
    mandelbrot_center_y = dy + 0.131825904

    # family_mix stays in [0.04, 0.22]: larger excursions need a different
    # camera chart and erase a deep Mandelbrot view's structure.
    family_mix = ((t * 0.24).sin() * 0.5 + 0.5) * 0.18 + 0.04

    # c = mu/2 - mu^2/4 parameterises the main cardioid; |mu| < 1 keeps the
    # Julia sets connected rather than dust.
    mu_x = (t * 0.31).cos() * 0.58
    mu_y = (t * 0.31).sin() * 0.58
    mu2_x = mu_x * mu_x - mu_y * mu_y
    mu2_y = mu_x * mu_y * 2.0
    julia_x = mu_x * 0.5 - mu2_x * 0.25
    julia_y = mu_y * 0.5 - mu2_y * 0.25

    # Preserve the target c-plane exactly under the family transform:
    # (1-mix)*pixel + mix*julia == mandelbrot_pixel.
    family_scale = family_mix * -1.0 + 1.0
    center_x = (mandelbrot_center_x - julia_x * family_mix) / family_scale
    center_y = (mandelbrot_center_y - julia_y * family_mix) / family_scale
    span = mandelbrot_span / family_scale
    drift = interest_network(unit_x=unit_x, unit_y=unit_y, interest=interest)
    return shade(quadratic_family(
        unit_x=unit_x + drift * 0.004,
        unit_y=unit_y + drift * -0.003,
        center_x=center_x, center_y=center_y, span=span,
        family_mix=family_mix, julia_x=julia_x, julia_y=julia_y,
    ))"""


NETWORK_SOURCE = """
def detail_network(travel, candidate):
    h0 = (travel * 0.18 + candidate * 0.72 - 0.31).tanh()
    h1 = (travel * -0.11 + candidate * 0.43 + 0.27).tanh()
    h2 = (travel * 0.07 + candidate * -0.61 + 0.08).tanh()
    h3 = (travel * -0.04 + candidate * 0.35 - 0.16).tanh()
    return (h0 * 0.52 + h1 * -0.33 + h2 * 0.41 + h3 * 0.28 + 0.5).tanh()


def score(travel, candidate):
    return detail_network(travel=travel, candidate=candidate)
"""


def compile_network_module():
    aot = compile_ast_aot(NETWORK_SOURCE, "score", {"travel": np.zeros(3), "candidate": np.zeros(3)}, backend="c", remove_loops=True, unroll_limit=4096, precompile_only=True)
    program = getattr(aot.compiled_shell_program, "program", aot.compiled_shell_program)
    arithmetic = [step for step in program.steps if step.op_name in {"add", "mul", "tanh"}]
    selected = type(program)(version=program.version, feeds=program.feeds, steps=program.steps, outputs={"detail_score": arithmetic[-1].result_id}, state_in=program.state_in, meta=program.meta, extras=program.extras)
    module = emit_wasm_module(selected, name="detail_network", dtype="float64")
    if not module.complete:
        raise SystemExit(module.shortfall_report())
    return module

def _bind_expressions(api) -> dict:
    """Attach each feed expression to the parameter the module declares.

    The page supplies the sampling grid, the clock, and the routed interest
    signal. Everything else -- where the camera is, how deep, which member of
    the family -- is computed inside the compiled program.
    """

    mapping = api.to_mapping()
    entry = next(
        (e for e in mapping["entry_points"] if e["name"] == mapping["entry"]),
        mapping["entry_points"][0],
    )
    inputs = [p["name"] for p in entry["parameters"] if p["role"] == "input"]

    known = {
        # normalized_plane, exactly: y spans [-0.5, 0.5] and x spans that
        # times the aspect ratio. The camera computes a span and then places
        # the view assuming the grid covers it -- so the grid multiplier IS
        # the zoom level, and 2.6 here silently showed 2.6x more plane than
        # the span the camera had chosen. Every lateral excursion was then
        # 2.6x smaller against the frame, and the deep view was never deep.
        "unit_x": "(x/(w-1) - 0.5) * (w/h)",
        "unit_y": "(y/(h-1) - 0.5)",
        "interest": "0.65 * Math.sin(t * 0.09) + 0.35 * Math.cos(t * 0.04)",
    }
    expressions = {name: known[name] for name in inputs if name in known}
    # The clock is whichever input is left. Named "t" when the capture
    # resolved it, positional when it did not -- either way it is the one
    # remaining, so it is found rather than guessed at.
    remaining = [name for name in inputs if name not in expressions]
    if len(remaining) != 1:
        raise SystemExit(
            "expected exactly one unbound input for the clock; the module "
            f"declares {inputs!r} and {len(remaining)} are unaccounted for"
        )
    expressions[remaining[0]] = "t"
    return expressions


def _clock_name(api) -> str:
    """Which declared input carries the clock.

    The feedback network advances a travel value and the shell writes it into
    whichever feed this names, so it has to agree with _bind_expressions or
    the network's dwell speed drives nothing.
    """

    expressions = _bind_expressions(api)
    return next(name for name, body in expressions.items() if body == "t")


def _channel_outputs(program) -> dict:
    """The three colour channels shade() computed, in order.

    A scalar operand is not an attribute: ``x.maximum(0.0)`` records a
    ``tensor_from_list`` step holding ``(0.0,)`` and then a ``maximum`` whose
    ``input_ids`` name it. Each channel ends on exactly that clamp, and it is
    the last thing the kernel does, so the final three are red, green, blue.
    """

    produced = {step.result_id: step for step in program.steps}

    def clamps_to_zero(step) -> bool:
        if step.op_name != "maximum" or len(step.input_ids) != 2:
            return False
        operand = produced.get(step.input_ids[1])
        return (
            operand is not None
            and operand.op_name == "tensor_from_list"
            and tuple(operand.attrs.get("values", ())) == (0.0,)
        )

    markers = [step for step in program.steps if clamps_to_zero(step)]
    if len(markers) < 3:
        raise SystemExit(
            f"expected three channel clamps in the program, found {len(markers)}"
        )
    red, green, blue = markers[-3:]
    return {"red": red.result_id, "green": green.result_id, "blue": blue.result_id}


def build(destination: Path) -> Path:
    channel = TelemetryChannel(name="homepage")
    network_module = compile_network_module()
    probe = {name: np.zeros(4) for name in (
        "unit_x", "unit_y", "t", "interest",
    )}

    with channel.stepped("building the homepage", 5, path="build") as advance:
        with channel.timed("parse + build_from_ast", path="frontend"):
            graph = ProcessGraph(materialize_memory=False)
            with contextlib.redirect_stdout(io.StringIO()):
                graph.build_from_ast(ast.parse(KERNEL))
            reduce_abstract_tensor_topology(graph)
        channel.log("process graph built", path="frontend",
                    nodes=graph.G.number_of_nodes(),
                    edges=graph.G.number_of_edges())
        advance("graph")

        with channel.timed("AOT lowering, loops unrolled", path="aot"):
            aot = compile_ast_aot(
                KERNEL, "render", probe, backend="c",
                remove_loops=True, unroll_limit=4096, precompile_only=True,
            )
        program = getattr(
            aot.compiled_shell_program, "program", aot.compiled_shell_program
        )
        channel.log("fused program", path="aot", steps=len(program.steps),
                    feeds=len(program.feeds), outputs=len(program.outputs))
        advance("lowered")

        # A capture names every observed value as an output. Saying which one
        # is the result is what lets everything else be pruned.
        adds = [s for s in program.steps if s.op_name == "add"]
        wanted = type(program)(
            version=program.version, feeds=program.feeds, steps=program.steps,
            outputs=_channel_outputs(program), state_in=program.state_in,
            meta=program.meta, extras=program.extras,
        )
        channel.log("selected the escape count", path="aot",
                    live_steps=len(required_steps(wanted)),
                    dead_steps=len(program.steps) - len(required_steps(wanted)))
        advance("pruned")

        with channel.timed("segmented WebAssembly emission and assembly", path="wasm"):
            specs = partition_reduced_program(
                wanted,
                chunk_size=WASM_REGION_STEPS,
                owner_name=aot.entrypoint,
            )
            modules = emit_class_modules(
                specs,
                dtype="float64",
                link_calls=False,
            )
            incomplete = [
                (spec, modules[spec.index])
                for spec in specs
                if not modules[spec.index].complete
            ]
            if incomplete:
                raise SystemExit("\n".join(
                    module.shortfall_report()
                    for _spec, module in incomplete
                ))
            api = describe_process_graph_api(
                specs,
                modules,
                wanted,
                entrypoint=aot.entrypoint,
            )
            class_graph = build_embedded_class_graph(
                specs,
                modules,
                wanted,
                entrypoint=aot.entrypoint,
                embed_binaries=False,
                module_dir=WASM_MODULE_DIR,
            )
            segmented_source = "\n\n".join(
                modules[spec.index].source for spec in specs
            )
            wasm_bytes = sum(
                len(modules[spec.index].binary) for spec in specs
            )
        channel.profile("assembled", path="wasm", nanoseconds=0,
                        bytes=wasm_bytes, regions=len(specs))
        advance("wasm")

        with channel.timed("emitting every backend", path="sources"):
            sources = collect_backend_sources(
                aot,
                channel=channel,
                wasm_source=segmented_source,
                program=wanted,
            )
        advance("sources")

    shell = emit_html_shell(
        api,
        source=segmented_source,
        class_graph=class_graph,
        name="index",
        telemetry=channel,
        process_graph=summarize_process_graph(graph),
        origin_source=KERNEL,
        backend_sources=sources,
        network_manifest={
            "name": "Mandelbrot future-detail controller",
            "module": {"api": network_module.api.to_mapping(), "wasm_base64": base64.b64encode(network_module.binary).decode("ascii")},
            "feedback": {"candidate_offsets": [0.0, 0.45, 0.9], "fps": 120, "render_fps": 24, "travel_feed": _clock_name(api)},
            "routes": [{"feed": _clock_name(api), "label": "network-guided travel", "effect": "future detail scores → dwell speed → camera clock → live frame"}],
        },
        # t is the frame number, so leaving "repeat" at 0 (continuous) makes
        # the view drift instead of recomputing one picture forever.
        # The zoom ping-pongs over a 160-frame cycle instead of descending
        # forever. Left unbounded it reaches a scale where every sample in
        # the window has the same escape count, and the picture goes flat --
        # correct arithmetic, nothing to look at.
        # The camera is a parameter now, so these drive it rather than
        # rebuilding the geometry per pixel. unit_x/unit_y stay the fixed
        # normalized plane; everything else is one number broadcast across
        # the grid, which is why the view can wander and dive without the
        # module ever being recompiled.
        #
        # The dive is exponential and runs to ~1e-11, which is deep enough
        # that the boundary keeps opening into new structure and still short
        # of the ~1e-15 where float64 starts quantising the plane into
        # visible blocks. It breathes back out rather than descending
        # forever, because at the bottom there is nothing left to resolve.
        # The page supplies the grid and the clock. The trajectory -- centre,
        # span, family blend, Julia constant -- is computed inside the
        # compiled module from t, so none of the algorithm runs here.
        # Feeds are named after the source parameters now -- the program
        # records which binding each came from, so the descriptor says
        # unit_x rather than feed0. The page supplies the grid and the
        # clock; the trajectory is computed inside the module.
        # Keyed by the names the module actually declares, read back from
        # its own descriptor rather than assumed. Most feeds resolve to their
        # source parameter; any that does not falls back to a positional name,
        # and the clock is matched by elimination rather than by guessing
        # which positional slot it landed in. If naming improves later this
        # keeps working untouched.
        feed_expressions=_bind_expressions(api),
        build_parameters={
            "iterations (unrolled)": ITERATIONS,
            "camera": "animated_camera + dream_parameters, in-kernel",
            "palette": "mandelbrot_jpeg_planes, in-kernel",
            "family": "continuous Mandelbrot <-> Julia, in-kernel",
            "camera": "computed in the module from t via baked sin/cos/exp2",
            "deepest span": "1.6 * 2^-36",
            "steps": len(required_steps(wanted)),
            "WASM regions": len(specs),
            "region step cap": WASM_REGION_STEPS,
            "wasm bytes": wasm_bytes,
            "interest model": "frozen 3→2→1 tanh network",
        },
        default_width=256,
        default_height=256,
    )
    written = shell.write(destination)
    region_directory = destination / WASM_MODULE_DIR
    region_directory.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (region_directory / f"{spec.module_name}.wasm").write_bytes(
            modules[spec.index].binary
        )
    served = len(sources.available())
    print(f"{served}/{len(sources.sources)} backends emitted this program")
    for entry in sources.sources:
        state = "ok " if entry.available else "n/a"
        print(f"  {entry.title:16} {state} {entry.lines:6} lines"
              f"  {entry.reason[:60]}")
    print(f"wrote {written} ({written.stat().st_size // 1024} KB)")
    print(f"wrote {len(specs)} lazy WASM regions to {region_directory}")
    return written


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
