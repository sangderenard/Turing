"""Plan or build the compiler's versioned outer ProcessGraph partition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.compiler.whole_program_compiler_bootstrap import (
    build_whole_program_compiler,
    plan_whole_program_compiler,
)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the ProcessGraph-to-AOT stage through the canonical "
            "partitioned source compiler"
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
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="maximum workers; auto mode also limits this to resolved-plan width",
    )
    parser.add_argument(
        "--source-compilation", choices=("auto", "parallel"), default="auto",
        help="resolved-graph unit scheduling policy (default: auto)",
    )
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
        "--worker-elastic-available-percent", type=int, default=25,
        help=(
            "percentage of signed system headroom above or below the system "
            "reserve applied to a compressed worker's temporary decoded "
            "front (default: 25)"
        ),
    )
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

    from src.compiler.shell_telemetry import compile_progress

    display = compile_progress(
        None,
        prefix="compiler-bootstrap: ",
    )
    with display:
        product = build_whole_program_compiler(
            arguments.output,
            jobs=arguments.jobs,
            source_compilation=arguments.source_compilation,
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
    print(
        f"outer partition {product.boundary['name']} "
        f"ended at {product.product.get('status')}; "
        f"manifest: {product.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
