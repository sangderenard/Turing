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

from .precompiled_graph import (
    LiveEvolutionEventBuffer,
    run_evolution_metagraph,
    run_precompiled_graph,
)


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


def live_process_graph_from_sympy(expression_text: str):
    """Return a live accessor populated through canonical SymPy ingestion."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False, source_language="sympy")
    accessor = graph.graph_accessor()

    def compile_after_window_opens() -> None:
        import sympy

        time.sleep(0.6)
        graph.build_from_expression(
            sympy.sympify(expression_text, evaluate=False)
        )

    worker = threading.Thread(
        target=compile_after_window_opens,
        name="turing-live-sympy-process-graph-build",
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

    metagraph = EvolutionMetaGraph()
    compiler_yield = max(0.0, float(config["compiler_yield_ms"])) / 1000.0

    def emit_event(event: Any) -> None:
        event_queue.put(event)
        if compiler_yield:
            # Open-loop resource sharing only: no visual backlog, physics
            # state, display frame, or acknowledgement is consulted here.
            stop_event.wait(compiler_yield)

    metagraph.subscribe(emit_event)
    result = ("stopped", "")
    try:
        if not stop_event.is_set():
            if config.get("input_kind") == "sympy":
                from ..compiler.autogenesis import compile_sympy_autogenesis

                compile_sympy_autogenesis(
                    config["source_text"],
                    metagraph=metagraph,
                    schedule=config["schedule_mode"],
                    final_target=config["final_target"],
                    pi_solver=config["sympy_pi_solver"],
                    pi_epsilon=config["sympy_pi_epsilon"],
                )
            else:
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
            result = ("ok", "")
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[translation-compiler-error] {message}", flush=True)
        result = ("error", message)
    finally:
        result_queue.put(result)
        event_queue.put(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Animate any supported compiler IR package with the "
            "CUDA-resident BoundSpring and asynchronous OpenGL surface"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--package", help="trusted .json/.pkl/Nodus GraphIR file")
    source.add_argument("--source", help="Python source file to ingest live")
    source.add_argument(
        "--sympy",
        metavar="EXPRESSION",
        help="compile this SymPy expression string through ProcessGraph ingestion",
    )
    parser.add_argument(
        "--entrypoint",
        help="compile this function through the full evolution recorder",
    )
    parser.add_argument(
        "--sympy-pi-solver",
        choices=("literal", "machin", "reject"),
        default="literal",
        help=(
            "pi materialization policy for SymPy ingestion: native f64 "
            "literal, error-bounded Machin series, or reject"
        ),
    )
    parser.add_argument(
        "--sympy-pi-epsilon",
        type=float,
        default=None,
        help="absolute error target for --sympy-pi-solver machin",
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
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument(
        "--release-hz",
        type=float,
        default=30.0,
        help="deprecated compatibility option; compiler events are now applied asynchronously",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=240.0,
        help="independent CUDA FluxSpring dispatch rate",
    )
    parser.add_argument(
        "--physics-device",
        default="cuda",
        help="Torch device for resident spring state (default: cuda; no implicit CPU fallback)",
    )
    parser.add_argument(
        "--spring-node-capacity",
        type=int,
        default=1_048_576,
        help="preloaded resident GPU node slots (default: 1,048,576)",
    )
    parser.add_argument(
        "--spring-edge-capacity",
        type=int,
        default=2_097_152,
        help="preloaded resident GPU edge slots (default: 2,097,152)",
    )
    parser.add_argument(
        "--spring-damping",
        type=float,
        default=0.902,
        help="continuous velocity damping coefficient (FluxSpring default: 0.902)",
    )
    parser.add_argument(
        "--spring-stiffness",
        type=float,
        default=64.0,
        help="edge spring stiffness (visibility-balanced default: 64.0)",
    )
    parser.add_argument(
        "--spring-repulsion-strength",
        type=float,
        default=0.005,
        help="node repulsion coefficient (visibility-balanced default: 0.005)",
    )
    parser.add_argument(
        "--spring-edge-pulse-mode",
        choices=("compiler-schedule", "off"),
        default="compiler-schedule",
        help="replay compiler-authored schedule groups through physical edges",
    )
    parser.add_argument(
        "--process-schedule-mode",
        "--spring-schedule-mode",
        dest="process_schedule_mode",
        choices=("asap", "alap"),
        default="asap",
        help="authoritative ProcessGraph scheduler policy (default: asap)",
    )
    parser.add_argument(
        "--spring-schedule-hz",
        type=float,
        default=3.0,
        help="compiler schedule replay groups per second",
    )
    parser.add_argument(
        "--spring-schedule-contraction",
        type=float,
        default=0.50,
        help=(
            "peak fractional rest-length reduction during the schedule "
            "envelope, clamped below 1 (default: 0.50)"
        ),
    )
    parser.add_argument(
        "--spring-contraction-response-hz",
        type=float,
        default=12.0,
        help=(
            "mechanical rest-length envelope follow/release rate, independent "
            "of schedule-group cadence (default: 12.0)"
        ),
    )
    parser.add_argument(
        "--spring-schedule-yank-impulse",
        type=float,
        default=0.0,
        help=(
            "frequency-normalized smooth inward force impulse per active "
            "edge envelope; try 0.04-0.10 (default: off)"
        ),
    )
    parser.add_argument(
        "--spring-boundary-radius",
        type=float,
        default=0.0,
        help=(
            "optional radial physics boundary; 0 disables the center-distance "
            "outer ring (default: disabled)"
        ),
    )
    parser.add_argument(
        "--spring-target-max-velocity",
        type=float,
        default=100.0,
        help=(
            "device-side target for previous-state maximum velocity; "
            "0 disables adaptive physics dt (default: 100)"
        ),
    )
    parser.add_argument(
        "--spring-min-dt-scale",
        type=float,
        default=0.1,
        help="minimum velocity-normalized physics dt multiplier",
    )
    parser.add_argument(
        "--spring-max-dt-scale",
        type=float,
        default=2.0,
        help="maximum velocity-normalized physics dt multiplier",
    )
    parser.add_argument(
        "--compiler-yield-ms",
        type=float,
        default=0.0,
        help=(
            "optional fixed delay after each compiler event to leave CPU time "
            "for physics/rendering; open-loop and never based on visual state"
        ),
    )
    parser.add_argument(
        "--max-event-backlog",
        type=int,
        default=256,
        help="deprecated compatibility value; event transport is always unbounded",
    )
    parser.add_argument(
        "--event-trace",
        action="store_true",
        help="print one summary for each drained compiler-event batch",
    )
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--report-hz", type=float, default=1.0)
    parser.add_argument(
        "--growth-depth-limit", type=int, default=0,
        help="deprecated compatibility option; compilation is never depth-clamped",
    )
    parser.add_argument(
        "--growth-height-limit", type=int, default=0,
        help="deprecated compatibility option; compilation is never height-clamped",
    )
    parser.add_argument(
        "--growth-branch-limit", type=int, default=0,
        help="deprecated compatibility option; compilation is never size-clamped",
    )
    parser.add_argument(
        "--growth-limit-boost",
        type=float,
        default=1.0,
        help="deprecated compatibility option; no growth ceiling is applied",
    )
    parser.add_argument(
        "--growth-restarts",
        type=int,
        default=4,
        help="deprecated compatibility option; visualization never restarts compilation",
    )
    parser.add_argument(
        "--growth-restart-wait",
        type=float,
        default=300.0,
        help="deprecated compatibility option; compiler never waits for visualization",
    )
    args = parser.parse_args(argv)

    worker = None
    relay = None
    visual_event_inbox = None
    worker_errors: list[BaseException] = []
    compiler_status = {"state": "not-started", "message": ""}
    if args.package:
        package = load_ir_package(args.package)
        runner = run_precompiled_graph
    else:
        source_text = (
            str(args.sympy)
            if args.sympy is not None
            else (
                Path(args.source).read_text(encoding="utf-8")
                if args.source
                else DEMO_SOURCE
            )
        )
        if args.topology_only:
            if args.sympy is not None:
                package, worker = live_process_graph_from_sympy(source_text)
            else:
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
            entrypoint = (
                None
                if args.sympy is not None
                else args.entrypoint
                if args.source
                else (args.entrypoint or "spectral_route")
            )
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
            # Queue(maxsize=0) hands events to multiprocessing's feeder without
            # waiting for physics, rendering, or the UI.  The visual inbox is
            # likewise unbounded: this path has no reverse flow-control edge.
            event_queue = process_context.Queue()
            result_queue = process_context.Queue(maxsize=1)
            visual_event_inbox = LiveEvolutionEventBuffer(
                args.max_event_backlog,
            )
            visual_event_inbox.activate()
            worker = process_context.Process(
                target=_isolated_autogenesis_worker,
                args=(event_queue, result_queue, stop_cascade, {
                    "source_text": source_text,
                    "input_kind": "sympy" if args.sympy is not None else "python",
                    "entrypoint": entrypoint,
                    "feeds": feeds,
                    "final_target": (
                        None if args.final_target.lower() == "none"
                        else args.final_target
                    ),
                    "boundary_namespace": args.boundary_namespace,
                    "source_language": args.source_language,
                    "extraction_contract": args.extraction_contract,
                    "compiler_yield_ms": args.compiler_yield_ms,
                    "schedule_mode": args.process_schedule_mode,
                    "sympy_pi_solver": args.sympy_pi_solver,
                    "sympy_pi_epsilon": args.sympy_pi_epsilon,
                }),
                name="turing-autogenesis-compiler",
                daemon=True,
            )
            worker.start()
            compiler_status["state"] = "compiling"

            def relay_compiler_events() -> None:
                while True:
                    event = event_queue.get()
                    if event is None:
                        break
                    visual_event_inbox.publish(event)
                    # The evolution projector is the sole visualization-side
                    # consumer. Rebuilding a second, unused EvolutionMetaGraph
                    # here doubles Python graph work and competes for the UI
                    # process GIL without contributing topology or telemetry.
                status, message = result_queue.get()
                compiler_status["state"] = (
                    "complete" if status == "ok" else status
                )
                compiler_status["message"] = message
                print(
                    "[translation-compiler-status] "
                    + compiler_status["state"]
                    + ((": " + message) if message else ""),
                    flush=True,
                )
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
            physics_hz=args.physics_hz,
            max_event_backlog=args.max_event_backlog,
            event_trace=args.event_trace,
            event_inbox=visual_event_inbox,
            spring_node_capacity=args.spring_node_capacity,
            spring_edge_capacity=args.spring_edge_capacity,
            physics_device=args.physics_device,
            spring_damping=args.spring_damping,
            spring_stiffness=args.spring_stiffness,
            spring_repulsion_strength=args.spring_repulsion_strength,
            spring_edge_pulse_mode=args.spring_edge_pulse_mode,
            spring_schedule_hz=args.spring_schedule_hz,
            spring_schedule_contraction=args.spring_schedule_contraction,
            spring_contraction_response_hz=(
                args.spring_contraction_response_hz
            ),
            spring_schedule_yank_impulse=args.spring_schedule_yank_impulse,
            spring_boundary_radius=args.spring_boundary_radius,
            spring_target_max_velocity=args.spring_target_max_velocity,
            spring_min_dt_scale=args.spring_min_dt_scale,
            spring_max_dt_scale=args.spring_max_dt_scale,
            compiler_status=compiler_status,
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
        if hasattr(worker, "kill") and worker.is_alive():
            worker.kill()
            worker.join(timeout=1.0)
    if relay is not None:
        relay.join(timeout=1.0)
    if visual_event_inbox is not None:
        visual_event_inbox.close()
    for process_queue_name in ("event_queue", "result_queue"):
        process_queue = locals().get(process_queue_name)
        if process_queue is not None:
            try:
                process_queue.cancel_join_thread()
                process_queue.close()
            except Exception:
                pass
    return 2 if worker_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
