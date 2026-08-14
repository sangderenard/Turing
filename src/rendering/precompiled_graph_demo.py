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


def live_process_graph_from_source(
    source: str,
    *,
    boundary_namespace: Any = None,
    source_language: str = "python",
):
    """Return a lock accessor while AST ingestion mutates its ProcessGraph."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(
        materialize_memory=False,
        boundary_namespace=boundary_namespace,
        source_language=source_language,
    )
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


def _isolated_autogenesis_worker(
    event_queue: Any,
    result_queue: Any,
    stop_event: Any,
    config: dict[str, Any],
) -> None:
    """Compile in another interpreter and stream authoritative events out."""

    from ..compiler.autogenesis import compile_source_autogenesis
    from ..compiler.evolution_metagraph import EvolutionMetaGraph
    from ..compiler.translation_growth_cascade import BoundaryRestartCascade
    from .precompiled_graph import ExpansionEmergencyClamp, ExpansionLimitExceeded

    metagraph = EvolutionMetaGraph()
    metagraph.subscribe(event_queue.put)
    cascade = BoundaryRestartCascade(
        Path(config["boundary_namespace"]),
        language=config["source_language"],
        max_restarts=config["growth_restarts"],
        wait_seconds=config["growth_restart_wait"],
        stop_event=stop_event,
        status_sink=lambda message: print(message, flush=True),
    )
    result = ("stopped", "")
    try:
        time.sleep(0.6)
        while not stop_event.is_set():
            clamp = ExpansionEmergencyClamp(
                max_depth=config["max_depth"],
                max_height=config["max_height"],
                max_nodes_per_branch=config["max_nodes_per_branch"],
            )
            unsubscribe_clamp = metagraph.subscribe(clamp)
            fingerprint = cascade.observing()
            try:
                compile_source_autogenesis(
                    config["source_text"],
                    config["entrypoint"],
                    config["feeds"],
                    metagraph=metagraph,
                    final_target=config["final_target"],
                    boundary_namespace=config["boundary_namespace"],
                    source_language=config["source_language"],
                    extraction_contract=config["extraction_contract"],
                )
                clamp.final_check()
                cascade.complete()
                result = ("ok", "")
                break
            except ExpansionLimitExceeded as exc:
                print(f"[translation-growth-clamp] {exc}", flush=True)
                cascade.flag(exc, fingerprint)
                if not cascade.wait_for_change(fingerprint):
                    result = (
                        "stopped" if stop_event.is_set() else "error",
                        str(exc),
                    )
                    break
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"[translation-compiler-error] {message}", flush=True)
                result = ("error", message)
                break
            finally:
                unsubscribe_clamp()
    finally:
        result_queue.put(result)
        event_queue.put(None)


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
        "--entrypoint",
        help="compile this function through the full evolution recorder",
    )
    parser.add_argument(
        "--feeds-json",
        help="JSON mapping of entrypoint feed names to scalar/array values",
    )
    parser.add_argument(
        "--final-target",
        default="none",
        help="optional final autogenesis target (default: stop after SSA)",
    )
    parser.add_argument(
        "--boundary-namespace",
        default=str(Path(__file__).resolve().parents[2] / "boundary_namespaces"),
        help="language/OOP boundary directory (default: project boundary_namespaces)",
    )
    parser.add_argument(
        "--source-language",
        default="python",
        help="first namespace directory below the boundary root",
    )
    parser.add_argument(
        "--extraction-contract",
        default=str(
            Path(__file__).resolve().parents[2]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        help="exhaustive YAML policy for source, host, native, and decompile choices",
    )
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
        default=30.0,
        help="exact ordered compiler events revealed per second (none are skipped)",
    )
    parser.add_argument(
        "--max-event-backlog",
        type=int,
        default=256,
        help="pause compilation after this many unrevealed live events",
    )
    parser.add_argument(
        "--event-trace",
        action="store_true",
        help="print each node/edge compiler event as the visualizer applies it",
    )
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report-hz", type=float, default=1.0)
    parser.add_argument("--growth-depth-limit", type=int, default=512)
    parser.add_argument("--growth-height-limit", type=int, default=512)
    parser.add_argument("--growth-branch-limit", type=int, default=50_000)
    parser.add_argument(
        "--growth-limit-boost",
        type=float,
        default=1.0,
        help="explicit multiplier for every emergency growth ceiling",
    )
    parser.add_argument(
        "--growth-restarts",
        type=int,
        default=4,
        help="restart attempts allowed after a boundary rule changes",
    )
    parser.add_argument(
        "--growth-restart-wait",
        type=float,
        default=300.0,
        help="seconds to keep physics live while waiting for a boundary edit",
    )
    args = parser.parse_args(argv)

    worker = None
    worker_errors: list[BaseException] = []
    if args.package:
        package = load_ir_package(args.package)
        runner = run_precompiled_graph
    else:
        source_text = (
            Path(args.source).read_text(encoding="utf-8")
            if args.source
            else DEMO_SOURCE
        )
        if args.topology_only or (args.source and not args.entrypoint):
            package, worker = live_process_graph_from_source(
                source_text,
                boundary_namespace=args.boundary_namespace,
                source_language=args.source_language,
            )
            runner = run_precompiled_graph
        else:
            from ..compiler.evolution_metagraph import EvolutionMetaGraph
            import multiprocessing
            import numpy as np

            package = EvolutionMetaGraph()
            if args.growth_limit_boost < 1.0:
                parser.error("--growth-limit-boost must be at least 1.0")
            boost = float(args.growth_limit_boost)
            entrypoint = args.entrypoint or "spectral_route"
            if args.feeds_json:
                raw_feeds = json.loads(
                    Path(args.feeds_json).read_text(encoding="utf-8")
                )
                feeds = {
                    name: np.asarray(value) if isinstance(value, list) else value
                    for name, value in raw_feeds.items()
                }
            elif args.source:
                feeds = {}
            else:
                feeds = {
                    "left": np.linspace(0.0, 1.0, 32),
                    "right": np.linspace(1.0, 0.0, 32),
                    "phase": np.full(32, 0.25),
                }

            process_context = multiprocessing.get_context("spawn")
            stop_cascade = process_context.Event()
            event_queue = process_context.Queue(maxsize=args.max_event_backlog)
            result_queue = process_context.Queue(maxsize=1)
            worker = process_context.Process(
                target=_isolated_autogenesis_worker,
                args=(event_queue, result_queue, stop_cascade, {
                    "source_text": source_text,
                    "entrypoint": entrypoint,
                    "feeds": feeds,
                    "final_target": (
                        None if args.final_target.lower() == "none"
                        else args.final_target
                    ),
                    "boundary_namespace": args.boundary_namespace,
                    "source_language": args.source_language,
                    "extraction_contract": args.extraction_contract,
                    "growth_restarts": args.growth_restarts,
                    "growth_restart_wait": args.growth_restart_wait,
                    "max_depth": max(1, round(args.growth_depth_limit * boost)),
                    "max_height": max(1, round(args.growth_height_limit * boost)),
                    "max_nodes_per_branch": max(
                        1, round(args.growth_branch_limit * boost)
                    ),
                }),
                name="turing-autogenesis-compiler",
                daemon=True,
            )
            worker.start()

            def relay_compiler_events() -> None:
                while True:
                    event = event_queue.get()
                    if event is None:
                        break
                    package.ingest_event(event)
                status, message = result_queue.get()
                if status == "error":
                    worker_errors.append(RuntimeError(message))

            relay = threading.Thread(
                target=relay_compiler_events,
                name="turing-autogenesis-event-relay",
                daemon=True,
            )
            relay.start()
            runner = run_evolution_metagraph

    runner_kwargs = dict(
        duration=args.duration,
        size=(args.width, args.height),
        fps=args.fps,
        release_hz=args.release_hz,
    )
    if runner is run_evolution_metagraph:
        runner_kwargs.update(
            top_k=args.top_k,
            report_hz=args.report_hz,
            max_event_backlog=args.max_event_backlog,
            event_trace=args.event_trace,
        )
    try:
        runner(package, **runner_kwargs)
    finally:
        if "stop_cascade" in locals():
            stop_cascade.set()
    if worker is not None:
        worker.join(timeout=1.0)
        if hasattr(worker, "terminate") and worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
    return 2 if worker_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
