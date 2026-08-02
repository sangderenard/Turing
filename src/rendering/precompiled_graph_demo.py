"""Command-line launcher for the reusable animated compiler graph surface."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
import threading
import time
from typing import Any

from .precompiled_graph import run_evolution_metagraph, run_precompiled_graph


DEMO_SOURCE = """
def spectral_route(left, right, phase):
    mixed = (left + right) * 0.5
    carrier = mixed.sin() + phase.cos()
    energy = carrier * carrier
    return energy / (1.0 + energy)
"""


def load_ir_package(path: str | Path) -> Any:
    """Load a trusted local JSON, pickle, or Nodus GraphIR package."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        # Pickle is intentionally restricted to this explicit trusted-local
        # CLI path; it is never accepted by the web publisher.
        with source.open("rb") as handle:
            return pickle.load(handle)
    text = source.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    return text


def live_process_graph_from_source(source: str):
    """Return a lock accessor while AST ingestion mutates its ProcessGraph."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    accessor = graph.graph_accessor()

    def compile_after_window_opens() -> None:
        time.sleep(0.6)
        graph.build_from_ast(ast.parse(source))

    worker = threading.Thread(
        target=compile_after_window_opens,
        name="turing-live-process-graph-build",
        daemon=True,
    )
    worker.start()
    return accessor, worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Animate any supported compiler IR package with the "
            "original FluxSpring-shader OpenGL graph surface"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--package", help="trusted .json/.pkl/Nodus GraphIR file")
    source.add_argument("--source", help="Python source file to ingest live")
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="show only live ProcessGraph construction, without IR handoffs",
    )
    parser.add_argument("--duration", type=float, default=float("inf"))
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--release-hz",
        type=float,
        default=6.0,
        help="buffered compiler topology revisions revealed per second",
    )
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    args = parser.parse_args(argv)

    worker = None
    if args.package:
        package = load_ir_package(args.package)
        runner = run_precompiled_graph
    else:
        source_text = (
            Path(args.source).read_text(encoding="utf-8")
            if args.source
            else DEMO_SOURCE
        )
        if args.topology_only or args.source:
            package, worker = live_process_graph_from_source(source_text)
            runner = run_precompiled_graph
        else:
            from ..compiler.autogenesis import compile_source_autogenesis
            from ..compiler.evolution_metagraph import EvolutionMetaGraph
            import numpy as np

            package = EvolutionMetaGraph()

            def compile_after_window_opens() -> None:
                time.sleep(0.6)
                result = compile_source_autogenesis(
                    source_text,
                    "spectral_route",
                    {
                        "left": np.linspace(0.0, 1.0, 32),
                        "right": np.linspace(1.0, 0.0, 32),
                        "phase": np.full(32, 0.25),
                    },
                    metagraph=package,
                )

            worker = threading.Thread(
                target=compile_after_window_opens,
                name="turing-autogenesis-compiler",
                daemon=True,
            )
            worker.start()
            runner = run_evolution_metagraph

    runner(
        package,
        duration=args.duration,
        size=(args.width, args.height),
        fps=args.fps,
        release_hz=args.release_hz,
    )
    if worker is not None:
        worker.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
