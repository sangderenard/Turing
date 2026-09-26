"""Plan or build the compiler as one whole-program DLL."""

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
            "Pursue the canonical compiler implementation from one root and "
            "link its complete repository-SSA call closure into one DLL"
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
        "--optimization",
        choices=("O0", "O1", "O2", "O3", "Os", "Oz"),
        default="O0",
    )
    arguments = parser.parse_args(argv)

    def report(message: str) -> None:
        print(message, flush=True)

    if arguments.plan_only:
        receipt = plan_whole_program_compiler(
            arguments.output,
            progress=report,
        )
        units = receipt["compilation_unit_plan"].get("units", ())
        print(
            f"planned one compiler program with {len(units)} compilation units; "
            f"receipt: {(arguments.output / 'plan.json').resolve()}",
            flush=True,
        )
        return 0

    product = build_whole_program_compiler(
        arguments.output,
        optimization=arguments.optimization,
        progress=report,
    )
    print(
        f"linked {len(product.module.functions)} compiler functions into "
        f"{product.library}; manifest: {product.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
