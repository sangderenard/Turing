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
from src.compiler.wasm_html_shell import emit_html_shell
from src.transmogrifier.graph.graph_express2 import ProcessGraph

# Every iteration is unrolled into the emitted program, so this is the one
# number that decides how large it gets. The escape test is the break-out:
# once a point has escaped, the comparison stops incrementing, so running
# more iterations refines the boundary rather than changing what it means.
ITERATIONS = 160

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
    # The dive is exponential in time and runs to 2^-36 -- deep enough that
    # the boundary keeps opening into new structure, and short of the ~1e-15
    # where float64 quantises the plane into visible blocks. It breathes back
    # out rather than descending forever, because at the bottom there is
    # nothing left to resolve.
    breath = (t * 0.0125).cos() * -0.5 + 0.5
    # 2^-36 written as exp(-36*ln2*b): exp2 has no AbstractTensor method, and
    # folding ln2 into the constant costs nothing. exp's table covers
    # [-30, 6], and this argument runs [-24.95, 0].
    span = (breath * -24.953210).exp() * 1.6
    # A slow wander along the seahorse valley, with a second, slower term so
    # the path does not simply repeat.
    center_x = (t * 0.021).sin() * 0.0022 + (t * 0.0071).cos() * 0.0009 - 0.743643887
    center_y = (t * 0.019).cos() * 0.0022 + (t * 0.0063).sin() * 0.0009 + 0.131825904
    # Mandelbrot for most of the cycle, leaning into the Julia family at the
    # top of each breath, where a fixed constant turns the same neighbourhood
    # into filigree.
    family_mix = (t * 0.00625).cos() * -0.5 + 0.5
    # The Julia constant walks the cardioid edge, where the interesting Julia
    # sets live.
    julia_x = (t * 0.013).cos() * 0.3943 - 0.35
    julia_y = (t * 0.013).sin() * 0.3943
    drift = interest_network(unit_x=unit_x, unit_y=unit_y, interest=interest)
    return quadratic_family(
        unit_x=unit_x + drift * 0.004,
        unit_y=unit_y + drift * -0.003,
        center_x=center_x, center_y=center_y, span=span,
        family_mix=family_mix, julia_x=julia_x, julia_y=julia_y,
    )"""


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
        "unit_x": "(x/(w-1) - 0.5) * 2.6",
        "unit_y": "(y/(h-1) - 0.5) * 2.6",
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
            outputs={"escape": adds[-1].result_id}, state_in=program.state_in,
            meta=program.meta, extras=program.extras,
        )
        channel.log("selected the escape count", path="aot",
                    live_steps=len(required_steps(wanted)),
                    dead_steps=len(program.steps) - len(required_steps(wanted)))
        advance("pruned")

        with channel.timed("WebAssembly emission and assembly", path="wasm"):
            module = emit_wasm_module(wanted, name="mandelbrot", dtype="float64")
        if not module.complete:
            raise SystemExit(module.shortfall_report())
        channel.profile("assembled", path="wasm", nanoseconds=0,
                        bytes=len(module.binary))
        advance("wasm")

        with channel.timed("emitting every backend", path="sources"):
            sources = collect_backend_sources(
                aot, channel=channel, wasm_source=module.source, program=wanted
            )
        advance("sources")

    shell = emit_html_shell(
        module.api,
        source=module.source,
        wasm_bytes=module.binary,
        name="index",
        telemetry=channel,
        process_graph=summarize_process_graph(graph),
        origin_source=KERNEL,
        backend_sources=sources,
        network_manifest={
            "name": "Mandelbrot future-detail controller",
            "module": {"api": network_module.api.to_mapping(), "wasm_base64": base64.b64encode(network_module.binary).decode("ascii")},
            "feedback": {"candidate_offsets": [0.0, 0.45, 0.9], "fps": 120, "render_fps": 24, "travel_feed": _clock_name(module.api)},
            "routes": [{"feed": _clock_name(module.api), "label": "network-guided travel", "effect": "future detail scores → dwell speed → camera clock → live frame"}],
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
        feed_expressions=_bind_expressions(module.api),
        build_parameters={
            "iterations (unrolled)": ITERATIONS,
            "family": "continuous Mandelbrot <-> Julia, in-kernel",
            "camera": "computed in the module from t via baked sin/cos/exp2",
            "deepest span": "1.6 * 2^-36",
            "steps": len(required_steps(wanted)),
            "wasm bytes": len(module.binary),
            "interest model": "frozen 3→2→1 tanh network",
        },
        default_width=256,
        default_height=256,
    )
    written = shell.write(destination)
    served = len(sources.available())
    print(f"{served}/{len(sources.sources)} backends emitted this program")
    for entry in sources.sources:
        state = "ok " if entry.available else "n/a"
        print(f"  {entry.title:16} {state} {entry.lines:6} lines"
              f"  {entry.reason[:60]}")
    print(f"wrote {written} ({written.stat().st_size // 1024} KB)")
    return written


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
