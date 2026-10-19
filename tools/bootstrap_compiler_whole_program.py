"""Plan or build the compiler's versioned outer ProcessGraph partition."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from src.compiler.whole_program_compiler_bootstrap import (
    bootstrap_provenance_core,
    build_whole_program_compiler,
    plan_whole_program_compiler,
)


def _provenance_schedule_options(arguments):
    max_total_gb = getattr(arguments, "max_total_gb", 10.0)
    max_worker_gb = getattr(arguments, "max_worker_gb", 10.0)
    return {
        "max_total_resident_bytes": (
            None if max_total_gb is None else int(max_total_gb * (1 << 30))
        ),
        "minimum_total_resident_bytes": int(
            getattr(arguments, "min_total_gb", 2.0) * (1 << 30)
        ),
        "system_reserve_bytes": int(
            getattr(arguments, "system_reserve_gb", 2.0) * (1 << 30)
        ),
        "worker_resident_reservation_bytes": int(
            getattr(arguments, "worker_reserve_gb", 2.0) * (1 << 30)
        ),
        "max_worker_memory_bytes": (
            None if max_worker_gb is None else int(max_worker_gb * (1 << 30))
        ),
        "worker_elastic_available_percent": getattr(
            arguments, "worker_elastic_available_percent", 75,
        ),
    }

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the ProcessGraph-to-AOT stage through the canonical "
            "source-list scheduler (auto inside each source)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/compiler-whole-program"),
    )
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument(
        "--plan-only",
        action="store_true",
        help="publish versioned boundary metadata without compiling",
    )
    stopping.add_argument(
        "--provenance-only",
        action="store_true",
        help=(
            "build, verify, install and restart into the native provenance core, "
            "then exit without submitting the outer compiler partition"
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="optional maximum source-list workers; inner auto remains implicit",
    )
    parser.add_argument(
        "--min-workers", "--minimum-workers", type=int, default=12,
        help="minimum admitted workers when that many independent jobs are ready (default: 12)",
    )
    parser.add_argument(
        "--source-compilation", choices=("auto", "serial", "parallel"), default="auto",
        help="auto inside each submitted source (default); legacy serial/parallel selects the source-list schedule",
    )
    parser.add_argument("--source-schedule", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--max-total-gb", type=float, default=10.0)
    parser.add_argument(
        "--min-total-gb", type=float, default=2.0,
        help="minimum adaptive aggregate worker cap (default: 2 GiB)",
    )
    parser.add_argument(
        "--system-reserve-gb", type=float, default=2.0,
        help="RAM left available to the rest of the system (default: 2 GiB)",
    )
    parser.add_argument("--worker-reserve-gb", type=float, default=2.0)
    parser.add_argument(
        "--max-worker-gb", type=float, default=10.0,
        help=(
            "per-worker memory-regulation/compression budget in GiB; "
            "crossing it does not terminate the compiler worker"
        ),
    )
    parser.add_argument(
        "--worker-elastic-available-percent", type=int, default=75,
        help=(
            "percentage of signed system headroom above or below the system "
            "reserve applied to a compressed worker's temporary decoded "
            "front (default: 75)"
        ),
    )
    parser.add_argument(
        "--continue-after-failure", action="store_true",
        help=(
            "admit every unit in each provenance stage and report all "
            "failed/partial units instead of stopping admission after the "
            "first observed failure"
        ),
    )
    for option in ("unroll-depth", "inline-depth", "custom-version-depth"):
        parser.add_argument("--" + option, type=int, default=0,
                            help="optional expansion nesting (default: 0, disabled)")
    arguments = parser.parse_args(argv)
    expansion_policy = {field: getattr(arguments, field) for field in
                        ("unroll_depth", "inline_depth", "custom_version_depth")}
    if any(value < 0 for value in expansion_policy.values()):
        parser.error("expansion depths must be non-negative")

    def report(message: str) -> None:
        print(message, flush=True)

    if arguments.plan_only:
        receipt = plan_whole_program_compiler(
            arguments.output,
            expansion_policy=expansion_policy,
            progress=report,
        )
        target = receipt["next_partition"]
        print(
            f"planned {len(receipt['boundaries'])} versioned boundaries; "
            f"first unit: {target['name']} "
            f"{target['version'][:12]}; "
            f"receipt: {(arguments.output / 'plan.json').resolve()}",
            flush=True,
        )
        return 0

    from src.compiler.compiler_bootstrap_runtime import (
        PROVENANCE_SOURCE_ONLY_ENV, ProvenanceCoreRestartRequired,
    )
    while True:
        try:
            # The provenance subprocess inherits this terminal and owns its
            # live compositor.  Do not open the outer compositor until it has
            # returned.
            plan, core = bootstrap_provenance_core(
                arguments.output,
                expansion_policy=expansion_policy,
                jobs=arguments.jobs,
                schedule_options=_provenance_schedule_options(arguments),
                stop_after_failure=not getattr(
                    arguments, "continue_after_failure", False
                ),
                progress=None,
            )
            break
        except ProvenanceCoreRestartRequired as restart:
            if not restart.requires_process_restart:
                continue
            # A new interpreter imports the complete native module before any
            # book or compiler worker exists; never replace a live Python
            # book.  Windows cannot overlay the current process reliably, so
            # keep the terminal attached while one fresh interpreter owns the
            # remainder of the bootstrap.
            environment = os.environ.copy()
            environment.pop(PROVENANCE_SOURCE_ONLY_ENV, None)
            command = [
                sys.executable, "-m", "tools.bootstrap_compiler_whole_program",
                *(argv if argv is not None else sys.argv[1:]),
            ]
            if os.name == "nt":
                return int(subprocess.run(command, env=environment).returncode)
            os.execve(sys.executable, command, environment)
        except subprocess.CalledProcessError as error:
            # The child already closed its display and printed the durable
            # failure rollup. Preserve its status without a parent traceback.
            return int(error.returncode or 1)

    if arguments.provenance_only:
        first = plan["next_partition"]
        print(
            f"provenance core {first['version'][:12]} selected from "
            f"{core.get('library', '<native module>')}; outer partition not started",
            flush=True,
        )
        return 0

    from src.compiler.shell_telemetry import compile_progress
    display = compile_progress(None, prefix="compiler-bootstrap: ")
    with display:
        return _build_and_report(arguments, expansion_policy, display)


def _build_and_report(arguments, expansion_policy, display):
    if arguments.provenance_only:
        plan, core = bootstrap_provenance_core(
            arguments.output,
            expansion_policy=expansion_policy,
            jobs=getattr(arguments, "jobs", None),
            schedule_options=_provenance_schedule_options(arguments),
            stop_after_failure=not getattr(
                arguments, "continue_after_failure", False
            ),
            progress=display.log,
        )
        first = plan["next_partition"]
        print(
            f"provenance core {first['version'][:12]} selected from "
            f"{core.get('library', '<native module>')}; outer partition not started",
            flush=True,
        )
        return 0

    product = build_whole_program_compiler(
        arguments.output,
        expansion_policy=expansion_policy,
        jobs=arguments.jobs,
        minimum_worker_jobs=arguments.min_workers,
        source_compilation=(arguments.source_schedule if arguments.source_compilation == "auto"
                            else arguments.source_compilation),
        max_total_resident_bytes=(
            None if arguments.max_total_gb is None
            else int(arguments.max_total_gb * (1 << 30))
        ),
        minimum_total_resident_bytes=int(
            arguments.min_total_gb * (1 << 30)
        ),
        system_reserve_bytes=int(
            arguments.system_reserve_gb * (1 << 30)
        ),
        adaptive_memory_ceiling=True,
        worker_resident_reservation_bytes=int(
            arguments.worker_reserve_gb * (1 << 30)
        ),
        max_worker_memory_bytes=(
            None if arguments.max_worker_gb is None
            else int(arguments.max_worker_gb * (1 << 30))
        ),
        worker_elastic_available_percent=(
            arguments.worker_elastic_available_percent
        ),
        progress=display.log,
    )
    if product.product.get("status") != "sealed":
        for record in product.product.get("units", ()):
            if record.get("native_verification_status") != "verified":
                display.log(f"FAILED {record['qualified_name']}: "
                    f"{record.get('error') or record.get('native_verification_reason') or record.get('native_failure') or record['status']}")
        display.close(failed=True)
    print(
        f"outer partition {product.boundary['name']} "
        f"ended at {product.product.get('status')}; "
        f"manifest: {product.manifest}",
        flush=True,
    )
    return 0 if product.product.get("status") == "sealed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
