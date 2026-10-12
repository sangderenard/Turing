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
        progress=report,
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
