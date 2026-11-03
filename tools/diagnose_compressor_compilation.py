"""Collect exact compressor unit failures with the incumbent isolated scheduler.

The source is ingested once through lower_ast_source_to_ssa, via the existing
partitioned-source planner. Failed units remain failures; this command neither
composes an incomplete program nor changes the compiler's acceptance gates.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_inventory(directory: Path, *, jobs: int = 12, memory_gb: float = 12):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.project_compilation_product import (
        compile_partitioned_source, compile_resolved_process_graph_plan,
    )

    started = time.perf_counter()
    last_plan = last_snapshot = 0.0
    directory = directory.resolve()

    def report(event):
        nonlocal last_plan, last_snapshot
        now = time.perf_counter()
        stage = event.get("stage")
        if stage == "source_plan" and now - last_plan >= 15:
            print(f"plan {now-started:.1f}s: {event.get('message', '')}", flush=True)
            last_plan = now
        elif stage == "resolved_worker_snapshot" and now - last_snapshot >= 15:
            running = event.get("running") or ()
            print(
                f"workers {now-started:.1f}s: running={len(running)} "
                f"pending={event.get('pending_count')} "
                f"completed={event.get('completed_count')} "
                f"active_units={tuple(item['unit_index'] for item in running)}",
                flush=True,
            )
            last_snapshot = now
        elif stage == "resolved_unit_start":
            print(
                f"start unit {event['unit_index']}: "
                f"{', '.join(event.get('qualified_names') or ())}", flush=True,
            )
        elif stage == "resolved_unit_finish":
            index = int(event["unit_index"])
            unit_dir = directory / "product" / "units" / f"unit_{index:03d}"
            path = unit_dir / "failure.json"
            if not path.exists():
                path = unit_dir / "unit.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            names = record.get("qualified_names") or record.get("unit", {}).get("qualified_names") or ()
            print(f"finish unit {index}: {record['status']} {', '.join(names)}", flush=True)
            if record.get("error"):
                print(f"  {record.get('error_type')}: {record['error']}", flush=True)
            print(f"  receipt: {path}", flush=True)

    plan_path = directory / "inputs" / "plan.json"
    if not plan_path.exists():
        source_module = importlib.import_module("src.common.tensors.compression.octet_codec")
        print(f"source: {source_module.__file__}; entry: compress_octets", flush=True)
        compile_partitioned_source(
            Path(source_module.__file__).read_text(encoding="utf-8"), directory,
            entrypoint="compress_octets", ssa_composition=True, plan_only=True,
            extraction_contract=ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml"),
            python_bindings=vars(source_module),
            source_options={"source_module": source_module.__name__,
                            "runtime_closure_only": True, "identity_log_level": "summary",
                            "tensor_ssa_reference": c_backend_repository_ssa_reference()},
            progress=report,
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    print(f"units={len(plan['units'])}; jobs={jobs}; stop_after_failure=False", flush=True)
    manifest = compile_resolved_process_graph_plan(
        directory / "resolved-process-graph.pkl", plan_path, directory / "product",
        jobs=jobs, schedule_mode="parallel", stop_after_failure=False,
        max_total_resident_bytes=int(memory_gb * (1 << 30)),
        minimum_total_resident_bytes=2 << 30, system_reserve_bytes=2 << 30,
        adaptive_memory_ceiling=True, worker_resident_reservation_bytes=512 << 20,
        max_worker_memory_bytes=4 << 30, progress=report,
    )
    print(
        f"finished in {time.perf_counter()-started:.2f}s: {manifest['counts']}; "
        f"peak_workers={manifest['concurrent_workers_peak']}; "
        f"pending={manifest['pending_unit_indices']}", flush=True,
    )
    print(f"manifest: {directory / 'product/manifest.json'}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory, or an unchanged frozen plan to resume")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--memory-gb", type=float, default=12)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.memory_gb < 2:
        parser.error("jobs must be positive and memory-gb at least 2")
    manifest = run_inventory(args.output, jobs=args.jobs, memory_gb=args.memory_gb)
    return int(any(unit.get("status") in {"failed", "partial"}
                   for unit in manifest["units"]) or bool(manifest["pending_unit_indices"]))


if __name__ == "__main__":
    raise SystemExit(main())
