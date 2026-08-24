"""Compile a Python project as isolated authored-call repository-SSA units."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Sequence

from src.compiler.project_compilation_product import (
    DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    compile_project_call,
    compile_project_bootstrap_creep,
    compile_project_product,
    compile_process_graph_creep,
    compile_process_graph_subdivision_integral,
    compile_process_graph_subdivision_plan,
    compile_resolved_process_graph_plan,
    compile_resolved_process_graph_unit,
    process_memory_bytes,
    publish_process_graph_subdivision_plan,
)


COMPILER_USAGE_TRACE_ROOT_ENV = "TURING_COMPILER_USAGE_TRACE_ROOT"


class _CompilerUsageProfiler:
    """Low-allocation call census for authored compiler functions."""

    def __init__(self, source_root: Path, destination: Path):
        self.source_root = source_root.resolve()
        self.destination = destination.resolve()
        self.prefix = os.path.normcase(str(self.source_root) + os.sep)
        self.code_keys = {}
        self.active = {}
        self.records = {}

    def _key(self, frame):
        code = frame.f_code
        cached = self.code_keys.get(code)
        if cached is not None:
            return cached
        filename = os.path.normcase(os.path.abspath(code.co_filename))
        key = (
            (Path(filename).resolve().as_posix(), str(code.co_qualname))
            if filename.startswith(self.prefix) else ()
        )
        self.code_keys[code] = key
        return key

    def __call__(self, frame, event, _argument):
        if event == "call":
            key = self._key(frame)
            if key:
                self.active[id(frame)] = (key, time.perf_counter())
        elif event == "return":
            active = self.active.pop(id(frame), None)
            if active is not None:
                key, started = active
                record = self.records.setdefault(key, [0, 0.0])
                record[0] += 1
                record[1] += time.perf_counter() - started

    def start(self) -> None:
        sys.setprofile(self)

    def finish(self) -> None:
        sys.setprofile(None)
        payload = {
            "schema": "turing.compiler-usage-trace.v1",
            "source_root": self.source_root.as_posix(),
            "process_id": os.getpid(),
            "records": [{
                "source": source,
                "qualified_name": qualified_name,
                "call_count": int(values[0]),
                "inclusive_seconds": float(values[1]),
            } for (source, qualified_name), values in sorted(
                self.records.items()
            )],
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_name(self.destination.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n",
        )
        os.replace(temporary, self.destination)


def _publish_bootstrap_runtime_state(
    destination: Path,
    state_provider,
) -> None:
    """Atomically publish activation telemetry to its dedicated receipt."""

    activation_receipt = {
        "schema": "turing.compiler-bootstrap-activation.v1",
        "products": list(state_provider()),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(activation_receipt, indent=2, sort_keys=True),
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, destination)


def _planned_unit_progress_writer(
    directory: Path,
    unit_index: int,
    unit: dict,
):
    """Persist the last internal phase of an isolated resolved-unit worker."""

    root = directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def report(message: str) -> None:
        text = str(message)
        phase = (
            "resolved-graph-load" if "loading post-reduction" in text else
            "deployment-selection" if "selecting complete" in text else
            "deployment-instantiation" if "instantiating complete" in text else
            "call-topology-validation" if "validating resolved" in text else
            "control-graph-planning" if "planning complete" in text else
            "region-precompile" if text.startswith("aot: lowering") else
            "repository-ssa-lowering" if "repository SSA" in text else
            "resolved-unit"
        )
        measured = process_memory_bytes(os.getpid())
        current = {
            "phase": phase,
            "message": text,
            "elapsed_seconds": time.perf_counter() - started,
            **({
                "resident_bytes": int(measured[0]),
                "private_bytes": int(measured[1]),
            } if measured is not None else {}),
        }
        payload = {
            "schema": "turing.resolved-process-graph-unit-progress.v1",
            "process_id": os.getpid(),
            "unit_index": int(unit_index),
            "qualified_names": list(unit.get("qualified_names") or ()),
            "current": current,
        }
        destination = root / "compile-progress.json"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n",
        )
        os.replace(temporary, destination)
        print(text, flush=True)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument(
        "--extraction-contract", type=Path,
        default=DEFAULT_PROJECT_EXTRACTION_CONTRACT,
        help=(
            "exhaustive callable-boundary contract (default: the repository "
            "program extraction contract)"
        ),
    )
    parser.add_argument("--linked-unit", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--linked-region", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-product", type=Path, default=None,
        help=(
            "reuse only receipt-matched, natively verified source regions "
            "from an earlier catalogue"
        ),
    )
    parser.add_argument(
        "--bootstrap-product", type=Path, action="append", default=[],
        help=(
            "re-prove and install a verified compiler product before this "
            "process works; isolated child workers inherit the same product"
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="concurrent isolated compiler workers (default: up to 4)",
    )
    parser.add_argument(
        "--max-total-gb", type=float, default=None,
        help="aggregate worker-RSS admission ceiling in GiB",
    )
    parser.add_argument(
        "--worker-reserve-gb", type=float, default=4.0,
        help="RAM reserved before admitting each worker (default: 4 GiB)",
    )
    parser.add_argument(
        "--max-worker-gb", type=float, default=4.0,
        help=(
            "terminate and record a unit before its committed/private memory "
            "exceeds this many GiB (default: 4; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--unit-timeout-seconds", type=float, default=300.0,
        help=(
            "terminate and record a unit after this many seconds "
            "(default: 300; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--emit-native", action="store_true",
        help="emit each complete dependency-linked unit as a shared library",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help=(
            "for exactly one entry, stop after post-reduction ProcessGraph "
            "division and publish its deterministic unit plan"
        ),
    )
    parser.add_argument(
        "--resolved-process-graph", type=Path,
        help="serialized post-reduction ProcessGraph produced by --plan-only",
    )
    parser.add_argument(
        "--process-graph-plan", type=Path,
        help="compilation-unit plan paired with --resolved-process-graph",
    )
    parser.add_argument(
        "--planned-unit", type=int,
        help="compile this zero-based resolved ProcessGraph unit only",
    )
    parser.add_argument(
        "--subdivision-plan", type=Path,
        help="structured deterministic child-integral plan from a refused unit",
    )
    parser.add_argument(
        "--subdivision-integral", type=int,
        help="compile this zero-based child integral from --subdivision-plan",
    )
    parser.add_argument(
        "--compile-subdivision-plan", action="store_true",
        help=(
            "crawl every child integral in --subdivision-plan using bounded "
            "isolated workers"
        ),
    )
    parser.add_argument(
        "--compile-resolved-plan", action="store_true",
        help=(
            "crawl every unit in a resolved ProcessGraph plan using bounded "
            "isolated workers"
        ),
    )
    parser.add_argument(
        "--creep-resolved-plan", action="store_true",
        help=(
            "autonomously crawl a resolved plan and every strictly deeper "
            "bounded subdivision until sealed or at an explicit fixed point"
        ),
    )
    parser.add_argument(
        "--max-subdivision-depth", type=int, default=32,
        help="maximum automatic creep depth (default: 32)",
    )
    parser.add_argument(
        "--creep-project", action="store_true",
        help=(
            "autonomously compile, verify, feed proven native products into "
            "later bounded passes, and stop at a durable fixed point"
        ),
    )
    parser.add_argument(
        "--max-bootstrap-rounds", type=int, default=16,
        help="maximum autonomous project bootstrap passes (default: 16)",
    )
    parser.add_argument(
        "--linked-planned-unit", action="append", default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    usage_profiler = None
    usage_root = os.environ.get(COMPILER_USAGE_TRACE_ROOT_ENV)
    if usage_root:
        usage_profiler = _CompilerUsageProfiler(
            Path(usage_root), arguments.output / "compiler-usage.json",
        )
        usage_profiler.start()
        atexit.register(usage_profiler.finish)

    from src.compiler.compiler_bootstrap_runtime import (
        activate_compiler_bootstrap_products,
        compiler_bootstrap_runtime_state,
        compiler_bootstrap_product_paths,
        set_compiler_bootstrap_products,
    )

    inherited_products = compiler_bootstrap_product_paths()
    selected_products = set_compiler_bootstrap_products((
        *inherited_products,
        *arguments.bootstrap_product,
    ))
    activations = activate_compiler_bootstrap_products(selected_products)
    if activations:
        arguments.output.mkdir(parents=True, exist_ok=True)
        activation_destination = (
            arguments.output / "bootstrap-activation.json"
        ).resolve()
        _publish_bootstrap_runtime_state(
            activation_destination, compiler_bootstrap_runtime_state,
        )
        # Pass the path as an atexit argument. Planned-unit failure handling
        # writes its own destination later; no mutable closure name can redirect
        # this receipt and overwrite failure.json.
        atexit.register(
            _publish_bootstrap_runtime_state,
            activation_destination,
            compiler_bootstrap_runtime_state,
        )
        print(json.dumps({
            "stage": "compiler_bootstrap_activation",
            "products": [item.to_mapping() for item in activations],
        }, sort_keys=True), flush=True)

    if arguments.creep_project:
        if arguments.source is None:
            parser.error("--creep-project requires --source")
        manifest = compile_project_bootstrap_creep(
            arguments.source,
            arguments.output,
            entries=arguments.entry or None,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * 1024 ** 3)
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * 1024 ** 3
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb == 0
                else int(arguments.max_worker_gb * 1024 ** 3)
            ),
            unit_timeout_seconds=(
                None if arguments.unit_timeout_seconds == 0
                else arguments.unit_timeout_seconds
            ),
            extraction_contract=arguments.extraction_contract,
            bootstrap_products=selected_products,
            seed_product=arguments.seed_product,
            max_rounds=arguments.max_bootstrap_rounds,
            progress=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True,
            ),
        )
        print(json.dumps({
            "status": manifest["status"],
            "rounds": len(manifest["rounds"]),
            "installed": len(manifest["installed_qualified_names"]),
            "manifest": str(arguments.output.resolve() / "manifest.json"),
        }, sort_keys=True), flush=True)
        return 0 if manifest["status"] == "sealed" else 1

    if arguments.creep_resolved_plan:
        if arguments.compile_resolved_plan or arguments.compile_subdivision_plan:
            parser.error(
                "--creep-resolved-plan cannot be combined with a one-level crawl"
            )
        if (
            arguments.resolved_process_graph is None
            or arguments.process_graph_plan is None
        ):
            parser.error(
                "--creep-resolved-plan requires --resolved-process-graph "
                "and --process-graph-plan"
            )
        manifest = compile_process_graph_creep(
            arguments.resolved_process_graph,
            arguments.process_graph_plan,
            arguments.output,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * 1024 ** 3)
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * 1024 ** 3
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb == 0
                else int(arguments.max_worker_gb * 1024 ** 3)
            ),
            unit_timeout_seconds=(
                None if arguments.unit_timeout_seconds == 0
                else arguments.unit_timeout_seconds
            ),
            max_subdivision_depth=arguments.max_subdivision_depth,
            bootstrap_products=selected_products,
            progress=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True,
            ),
        )
        print(json.dumps({
            "status": manifest["status"],
            "rounds": len(manifest["rounds"]),
            "verified_products": len(manifest["verified_products"]),
            "fixed_points": len(manifest["fixed_points"]),
            "manifest": str(arguments.output.resolve() / "manifest.json"),
        }, sort_keys=True), flush=True)
        return 0 if manifest["status"] == "sealed" else 1

    if arguments.compile_resolved_plan:
        if (
            arguments.resolved_process_graph is None
            or arguments.process_graph_plan is None
        ):
            parser.error(
                "--compile-resolved-plan requires --resolved-process-graph "
                "and --process-graph-plan"
            )
        manifest = compile_resolved_process_graph_plan(
            arguments.resolved_process_graph,
            arguments.process_graph_plan,
            arguments.output,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * 1024 ** 3)
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * 1024 ** 3
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb == 0
                else int(arguments.max_worker_gb * 1024 ** 3)
            ),
            unit_timeout_seconds=(
                None if arguments.unit_timeout_seconds == 0
                else arguments.unit_timeout_seconds
            ),
            progress=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True,
            ),
        )
        print(json.dumps(manifest["counts"], sort_keys=True), flush=True)
        return 0

    if arguments.compile_subdivision_plan:
        if arguments.subdivision_plan is None:
            parser.error(
                "--compile-subdivision-plan requires --subdivision-plan"
            )
        if arguments.subdivision_integral is not None:
            parser.error(
                "--compile-subdivision-plan cannot select one integral"
            )
        manifest = compile_process_graph_subdivision_plan(
            arguments.subdivision_plan,
            arguments.output,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * 1024 ** 3)
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * 1024 ** 3
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb == 0
                else int(arguments.max_worker_gb * 1024 ** 3)
            ),
            unit_timeout_seconds=(
                None if arguments.unit_timeout_seconds == 0
                else arguments.unit_timeout_seconds
            ),
            progress=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True,
            ),
        )
        print(json.dumps(manifest["counts"], sort_keys=True), flush=True)
        return 0

    subdivision_mode = (
        arguments.subdivision_plan is not None
        or arguments.subdivision_integral is not None
    )
    if subdivision_mode:
        if (
            arguments.subdivision_plan is None
            or arguments.subdivision_integral is None
        ):
            parser.error(
                "subdivision mode requires --subdivision-plan and "
                "--subdivision-integral"
            )
        subdivision_plan = json.loads(
            arguments.subdivision_plan.read_text(encoding="utf-8")
        )
        integrals = tuple(subdivision_plan.get("integrals") or ())
        selected_integral = (
            dict(integrals[arguments.subdivision_integral])
            if 0 <= int(arguments.subdivision_integral) < len(integrals)
            else {}
        )
        subdivision_progress = _planned_unit_progress_writer(
            arguments.output,
            int(arguments.subdivision_integral),
            selected_integral,
        )
        try:
            receipt = compile_process_graph_subdivision_integral(
                arguments.subdivision_plan,
                arguments.subdivision_integral,
                arguments.output,
                progress=subdivision_progress,
            )
        except Exception as error:
            failure = {
                "schema": "turing.process-graph-subdivision-failure.v1",
                "status": "failed",
                "integral_index": int(arguments.subdivision_integral),
                "integral": selected_integral,
                "qualified_names": list(
                    selected_integral.get("qualified_names") or ()
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            diagnostic = getattr(error, "to_failure_mapping", None)
            if callable(diagnostic):
                failure.update(dict(diagnostic()))
            if selected_integral.get("parent_unit_index") is not None:
                failure["unit_index"] = int(
                    selected_integral["parent_unit_index"]
                )
                failure["unit"] = {
                    "qualified_names": list(
                        selected_integral.get("qualified_names") or ()
                    ),
                    "function_references": list(
                        selected_integral.get("function_references") or ()
                    ),
                }
            progress_path = arguments.output / "compile-progress.json"
            if progress_path.is_file():
                try:
                    failure["stage"] = json.loads(
                        progress_path.read_text(encoding="utf-8")
                    ).get("current")
                except (OSError, TypeError, ValueError):
                    pass
            arguments.output.mkdir(parents=True, exist_ok=True)
            failure_destination = arguments.output / "failure.json"
            temporary = failure_destination.with_name(
                failure_destination.name + ".tmp"
            )
            temporary.write_text(
                json.dumps(failure, indent=2, sort_keys=True),
                encoding="utf-8", newline="\n",
            )
            os.replace(temporary, failure_destination)
            print(failure["traceback"], flush=True)
            return 1
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0

    planned_mode = any(value is not None for value in (
        arguments.resolved_process_graph,
        arguments.process_graph_plan,
        arguments.planned_unit,
    ))
    if planned_mode:
        if (
            arguments.resolved_process_graph is None
            or arguments.process_graph_plan is None
            or arguments.planned_unit is None
        ):
            parser.error(
                "planned-unit mode requires --resolved-process-graph, "
                "--process-graph-plan, and --planned-unit"
            )
        plan = json.loads(
            arguments.process_graph_plan.read_text(encoding="utf-8")
        )
        units = tuple(plan.get("units") or ())
        selected = (
            dict(units[arguments.planned_unit])
            if 0 <= int(arguments.planned_unit) < len(units) else {}
        )
        planned_progress = _planned_unit_progress_writer(
            arguments.output, int(arguments.planned_unit), selected,
        )
        linked_units = {
            int(record["unit_index"]): Path(record["root"])
            for payload in arguments.linked_planned_unit
            for record in (json.loads(payload),)
        }
        try:
            receipt = compile_resolved_process_graph_unit(
                arguments.resolved_process_graph,
                arguments.process_graph_plan,
                arguments.planned_unit,
                arguments.output,
                linked_units=linked_units,
                progress=planned_progress,
            )
        except Exception as error:
            failure = {
                "schema": "turing.resolved-process-graph-unit-failure.v1",
                "status": "failed",
                "unit_index": int(arguments.planned_unit),
                "unit": selected,
                "qualified_names": list(
                    selected.get("qualified_names") or ()
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            diagnostic = getattr(error, "to_failure_mapping", None)
            if callable(diagnostic):
                failure.update(dict(diagnostic()))
            arguments.output.mkdir(parents=True, exist_ok=True)
            destination = arguments.output / "failure.json"
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(
                json.dumps(failure, indent=2, sort_keys=True),
                encoding="utf-8", newline="\n",
            )
            os.replace(temporary, destination)
            publish_process_graph_subdivision_plan(
                arguments.output,
                (failure,),
                arguments.resolved_process_graph,
                arguments.process_graph_plan,
            )
            print(failure["traceback"], flush=True)
            return 1
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0

    if arguments.source is None:
        parser.error("--source is required except in planned-unit mode")

    if arguments.worker:
        if len(arguments.entry) != 1:
            parser.error("worker mode requires exactly one --entry")
        try:
            linked_units = {}
            for payload in arguments.linked_unit:
                record = json.loads(payload)
                linked_units[str(record["qualified_name"])] = (
                    Path(record["artifact"]), str(record["root"]),
                )
            linked_regions = {}
            for payload in arguments.linked_region:
                record = json.loads(payload)
                linked_regions[tuple(map(
                    str, record["identity_token_chain"]
                ))] = (
                    Path(record["artifact"]), Path(record["verification"]),
                )
            receipt = compile_project_call(
                arguments.source,
                arguments.entry[0],
                arguments.output,
                extraction_contract=arguments.extraction_contract,
                linked_units=linked_units,
                linked_regions=linked_regions,
                plan_only=arguments.plan_only,
                progress=lambda message: print(
                    f"[{arguments.entry[0]}] {message}", flush=True,
                ),
            )
        except Exception as error:
            progress_path = arguments.output / "compile-progress.json"
            failure_stage = None
            if progress_path.is_file():
                try:
                    failure_stage = json.loads(
                        progress_path.read_text(encoding="utf-8")
                    ).get("current")
                except (OSError, ValueError, TypeError):
                    failure_stage = None
            failure = {
                "schema": "turing.project-compilation-failure.v1",
                "qualified_name": arguments.entry[0],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                **({"stage": failure_stage} if failure_stage else {}),
            }
            diagnostic = getattr(error, "to_failure_mapping", None)
            if callable(diagnostic):
                failure.update(dict(diagnostic()))
            process_graph_plan = arguments.output / "process-graph-units.json"
            if process_graph_plan.is_file():
                failure["process_graph_unit_plan"] = process_graph_plan.name
            resolved_process_graph = (
                arguments.output / "resolved-process-graph.pkl"
            )
            if resolved_process_graph.is_file():
                failure["resolved_process_graph"] = resolved_process_graph.name
            arguments.output.mkdir(parents=True, exist_ok=True)
            destination = arguments.output / "failure.json"
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(
                json.dumps(failure, indent=2, sort_keys=True),
                encoding="utf-8", newline="\n",
            )
            os.replace(temporary, destination)
            if (
                process_graph_plan.is_file()
                and resolved_process_graph.is_file()
            ):
                publish_process_graph_subdivision_plan(
                    arguments.output,
                    (failure,),
                    resolved_process_graph,
                    process_graph_plan,
                )
            print(failure["traceback"], flush=True)
            return 1
        print(json.dumps({
            "qualified_name": receipt["qualified_name"],
            **({"artifact": receipt["artifact"]}
               if receipt.get("artifact") else {}),
            **({"process_graph_unit_plan": receipt["process_graph_unit_plan"]}
               if receipt.get("process_graph_unit_plan") else {}),
            "elapsed_seconds": receipt["elapsed_seconds"],
        }, sort_keys=True), flush=True)
        return 0

    if arguments.plan_only:
        if len(arguments.entry) != 1:
            parser.error("--plan-only requires exactly one --entry")
        receipt = compile_project_call(
            arguments.source,
            arguments.entry[0],
            arguments.output,
            extraction_contract=arguments.extraction_contract,
            plan_only=True,
            progress=lambda message: print(
                f"[{arguments.entry[0]}] {message}", flush=True,
            ),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0

    manifest = compile_project_product(
        arguments.source,
        arguments.output,
        entries=arguments.entry or None,
        jobs=arguments.jobs,
        max_total_resident_bytes=(
            None
            if arguments.max_total_gb is None
            else int(arguments.max_total_gb * 1024 ** 3)
        ),
        worker_resident_reservation_bytes=int(
            arguments.worker_reserve_gb * 1024 ** 3
        ),
        max_worker_memory_bytes=(
            None
            if arguments.max_worker_gb == 0
            else int(arguments.max_worker_gb * 1024 ** 3)
        ),
        unit_timeout_seconds=(
            None
            if arguments.unit_timeout_seconds == 0
            else arguments.unit_timeout_seconds
        ),
        extraction_contract=arguments.extraction_contract,
        emit_native=arguments.emit_native,
        seed_product=arguments.seed_product,
        progress=lambda event: print(
            json.dumps(event, sort_keys=True), flush=True,
        ),
    )
    completed = sum(
        unit["status"] == "complete" for unit in manifest["units"]
    )
    partial = sum(unit["status"] == "partial" for unit in manifest["units"])
    failed = sum(unit["status"] == "failed" for unit in manifest["units"])
    blocked = sum(unit["status"] == "blocked" for unit in manifest["units"])
    print(
        f"catalogue complete: {completed} linkable, {partial} partial, "
        f"{failed} failed, {blocked} blocked -> "
        f"{arguments.output.resolve() / 'manifest.json'}",
        flush=True,
    )
    return 1 if failed or partial or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
