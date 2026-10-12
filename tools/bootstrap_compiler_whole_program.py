"""Plan or build the compiler's versioned outer ProcessGraph partition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.compiler.whole_program_compiler_bootstrap import (
    build_whole_program_compiler,
    plan_whole_program_compiler,
)


def _worker_key(prefix: str, qualified_name: str, index: Any = "") -> str:
    return f"{prefix}:{qualified_name}:{index}"


def _show_worker(
    display,
    key: str,
    label: str,
    current: Mapping[str, Any],
    *,
    status: str = "running",
    rss_bytes: int = 0,
) -> None:
    compiler = dict(current.get("compiler") or {})
    if hasattr(display, "worker"):
        display.worker(
            key,
            label=label,
            status=status,
            stage_number=int(compiler.get("stage_number") or 0),
            stage_count=int(compiler.get("stage_count") or 0),
            stage_label=str(
                compiler.get("stage_label")
                or current.get("message")
                or "starting"
            ),
            substage=str(compiler.get("substage") or ""),
            elapsed_seconds=float(current.get("elapsed_seconds") or 0.0),
            rss_bytes=int(
                rss_bytes or current.get("resident_bytes") or 0
            ),
            stages=tuple(compiler.get("stages") or ()),
        )
        return
    display.count(
        key, 0, total=1, label=label,
        detail=str(current.get("message") or "starting"),
    )


def _report_bootstrap_event(display, event: Mapping[str, Any]) -> None:
    """Project all worker events onto the one parent-owned terminal frame."""

    stage = str(event.get("stage") or "")
    if stage == "bootstrap_creep_round_start":
        display.substage(
            f"bootstrap round {int(event.get('round') or 0) + 1}"
        )
        return
    if stage == "bootstrap_creep_worker":
        worker = dict(event.get("event") or {})
        worker_stage = str(worker.get("stage") or "")
        qualified_name = str(worker.get("qualified_name") or "worker")
        key = _worker_key("catalogue", qualified_name)
        if worker_stage == "unit_start":
            _show_worker(display, key, qualified_name, {}, status="starting")
        elif worker_stage == "worker_snapshot":
            display.count(
                "catalogue-total",
                int(worker.get("completed_count") or 0),
                total=int(worker.get("count") or 0),
                label="authored compiler units",
                detail=f"{len(worker.get('running') or ())} running",
            )
            for row_value in worker.get("running") or ():
                row = dict(row_value)
                name = str(row.get("qualified_name") or "worker")
                row_key = _worker_key("catalogue", name)
                current = dict(row.get("stage") or {})
                current["elapsed_seconds"] = float(
                    row.get("elapsed_seconds") or 0.0
                )
                resident = int(row.get("resident_bytes") or 0)
                _show_worker(
                    display, row_key, name, current,
                    status="running", rss_bytes=resident,
                )
        elif worker_stage == "worker_output":
            message = str(worker.get("message") or "")
            if message:
                display.log(f"[{qualified_name}] {message}")
        elif worker_stage == "unit_complete":
            status = str(worker.get("status") or "unknown")
            failure_stage = worker.get("failure_stage")
            current = dict(
                failure_stage if isinstance(failure_stage, Mapping)
                else worker.get("stage")
                if isinstance(worker.get("stage"), Mapping)
                else {}
            )
            current.setdefault(
                "elapsed_seconds", float(worker.get("elapsed_seconds") or 0.0)
            )
            _show_worker(
                display, key, qualified_name, current, status=status,
                rss_bytes=int(worker.get("peak_resident_bytes") or 0),
            )
            display.log(f"{qualified_name}: {status}")
        return
    if stage == "bootstrap_creep_subdivision":
        qualified_name = str(event.get("qualified_name") or "subdivision")
        nested = dict(event.get("event") or {})
        if str(nested.get("stage") or "") == "creep_worker":
            worker = dict(nested.get("event") or {})
            worker_stage = str(worker.get("stage") or "")
            unit_index = worker.get(
                "unit_index", worker.get("integral_index", ""),
            )
            key = _worker_key("subdivision", qualified_name, unit_index)
            if worker_stage in {
                "resolved_worker_snapshot", "subdivision_worker_snapshot",
            }:
                index_field = (
                    "unit_index" if worker_stage.startswith("resolved")
                    else "integral_index"
                )
                display.count(
                    f"{key}:total",
                    int(worker.get("completed_count") or 0),
                    total=int(worker.get("count") or 0),
                    label=qualified_name,
                    detail=f"{len(worker.get('running') or ())} subdivision workers",
                )
                for row_value in worker.get("running") or ():
                    row = dict(row_value)
                    row_index = row.get(index_field, "")
                    row_key = _worker_key(
                        "subdivision", qualified_name, row_index,
                    )
                    current = dict(row.get("stage") or {})
                    detail = str(current.get("message") or "running")
                    resident = int(row.get("resident_bytes") or 0)
                    if resident:
                        detail += f" | rss {resident / (1 << 30):.2f}GB"
                    display.count(
                        row_key, 0, total=1, label=qualified_name,
                        detail=detail,
                    )
            elif worker_stage.endswith("_start"):
                display.count(
                    key, 0, total=1, label=qualified_name,
                    detail=f"subdivision unit {unit_index}",
                )
            elif worker_stage.endswith(("_finish", "_failure")):
                status = str(worker.get("status") or "complete")
                display.count(
                    key, 1, total=1, label=qualified_name, detail=status,
                )
                display.count_close(key)
        elif str(nested.get("stage") or "") == "creep_round_finish":
            counts = dict(nested.get("counts") or {})
            display.count_close(
                f"{_worker_key('subdivision', qualified_name)}:total"
            )
            display.log(f"{qualified_name}: subdivision {counts}")
        return
    if stage == "bootstrap_creep_round_finish":
        display.count_close("catalogue-total")
        failures = tuple(event.get("continuation_failures") or ())
        if failures:
            display.log(
                "bootstrap round retained Python frontiers: "
                + ", ".join(
                    str(item.get("qualified_name") or "<unknown>")
                    for item in failures
                )
            )
        if hasattr(display, "worker_clear"):
            display.worker_clear()
        return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the compiler's existing native boundaries, then offer "
            "the complete ProcessGraph-to-AOT outer stage to bootstrap creep"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/compiler-whole-program"),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="stop after publishing the resolved compilation-unit plan",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-total-gb", type=float)
    parser.add_argument("--worker-reserve-gb", type=float, default=8.0)
    parser.add_argument("--max-worker-gb", type=float, default=12.0)
    parser.add_argument("--unit-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-rounds", type=int, default=16)
    arguments = parser.parse_args(argv)

    def report(message: str) -> None:
        print(message, flush=True)

    if arguments.plan_only:
        receipt = plan_whole_program_compiler(
            arguments.output,
            progress=report,
        )
        target = receipt["next_partition"]
        print(
            f"planned {len(receipt['boundaries'])} versioned boundaries; "
            f"next outer partition: {target['name']} "
            f"{target['version'][:12]}; "
            f"receipt: {(arguments.output / 'plan.json').resolve()}",
            flush=True,
        )
        return 0

    from src.compiler.shell_telemetry import CompileStage, compile_progress

    display = compile_progress(
        None,
        stages=(CompileStage("bootstrap", "compiler bootstrap"),),
        prefix="compiler-bootstrap: ",
    )
    with display:
        product = build_whole_program_compiler(
            arguments.output,
            jobs=arguments.jobs,
            max_total_resident_bytes=(
                None if arguments.max_total_gb is None
                else int(arguments.max_total_gb * (1 << 30))
            ),
            worker_resident_reservation_bytes=int(
                arguments.worker_reserve_gb * (1 << 30)
            ),
            max_worker_memory_bytes=(
                None if arguments.max_worker_gb is None
                else int(arguments.max_worker_gb * (1 << 30))
            ),
            unit_timeout_seconds=arguments.unit_timeout_seconds,
            max_rounds=arguments.max_rounds,
            progress=display.log,
            event_progress=lambda event: _report_bootstrap_event(display, event),
        )
    print(
        f"outer partition {product.boundary['name']} "
        f"ended at {product.product.get('status')}; "
        f"manifest: {product.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
