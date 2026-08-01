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
ITERATIONS = 48

KERNEL = f"""
def interest_network(cx, cy, interest):
    # Frozen network source enters the same AOT/WASM pipeline as the fractal.
    h0 = (cx * 0.83 + cy * -0.41 + interest * 0.72 + 0.15).tanh()
    h1 = (cx * -0.36 + cy * 0.91 + interest * 0.48 - 0.08).tanh()
    return (h0 * 0.62 + h1 * -0.57 + interest * 0.31).tanh()


def mandelbrot_escape(cx, cy):
    zx = cx * 0.0
    zy = cx * 0.0
    count = cx * 0.0
    for _ in range({ITERATIONS}):
        zx2 = zx * zx
        zy2 = zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        next_zx = zx2 - zy2 + cx
        next_zy = 2.0 * zx * zy + cy
        zx = next_zx
        zy = next_zy
    return count


def render(cx, cy, interest):
    recommendation = interest_network(cx=cx, cy=cy, interest=interest)
    return mandelbrot_escape(cx=cx + recommendation * 0.018, cy=cy + recommendation * -0.012)"""


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

def build(destination: Path) -> Path:
    channel = TelemetryChannel(name="homepage")
    network_module = compile_network_module()
    probe = {"cx": np.zeros(4), "cy": np.zeros(4), "interest": np.zeros(4)}

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
            "feedback": {"candidate_offsets": [0.0, 0.45, 0.9], "fps": 120, "render_fps": 24, "travel_feed": "feed2"},
            "routes": [{"feed": "feed2", "label": "network-guided travel", "effect": "future detail scores → dwell speed → live frame"}],
        },
        # t is the frame number, so leaving "repeat" at 0 (continuous) makes
        # the view drift instead of recomputing one picture forever.
        # The zoom ping-pongs over a 160-frame cycle instead of descending
        # forever. Left unbounded it reaches a scale where every sample in
        # the window has the same escape count, and the picture goes flat --
        # correct arithmetic, nothing to look at.
        feed_expressions={
            "feed0": "-0.743644 + (x/(w-1) - 0.5) * 3.0 * "
                     "Math.pow(0.955, 80 - Math.abs(80 - (t % 160)))",
            "feed1": "0.131826 + (y/(h-1) - 0.5) * 2.4 * "
                     "Math.pow(0.955, 80 - Math.abs(80 - (t % 160)))",
            "feed2": "0.65 * Math.sin(t * 0.09) + 0.35 * Math.cos(t * 0.04)",
        },
        build_parameters={
            "iterations (unrolled)": ITERATIONS,
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
