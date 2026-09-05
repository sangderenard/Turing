"""Build and optionally run the canonical Python balloon-tire native program."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.vehicle_python_compilation import (
    compile_balloon_tire_managed_python_native,
    compile_balloon_tire_python_native,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "balloon-tire-native",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--managed-dt",
        action="store_true",
        help="compile the balloon step inside the repository dt controller",
    )
    parser.add_argument("--window-duration", type=float, default=1.0 / 120.0)
    parser.add_argument("--dt-initial", type=float, default=1.0 / 360.0)
    parser.add_argument(
        "--backend", choices=("c", "fortran"), default="c",
    )
    parser.add_argument(
        "--optimization",
        choices=("O0", "O1", "O2", "O3", "Os", "Oz"),
        default="O2",
    )
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    arguments = parser.parse_args()

    progress = lambda message: print(message, flush=True)
    if arguments.managed_dt:
        if arguments.backend != "c":
            parser.error("--managed-dt currently requires --backend c")
        if arguments.trace:
            parser.error("--trace is not implemented for the managed C host")
        artifact = compile_balloon_tire_managed_python_native(
            arguments.output,
            batch_size=arguments.batch_size,
            window_duration=arguments.window_duration,
            dt_initial=arguments.dt_initial,
            optimization=arguments.optimization,
            progress=progress,
        )
    else:
        artifact = compile_balloon_tire_python_native(
            arguments.output,
            batch_size=arguments.batch_size,
            backend=arguments.backend,
            optimization=arguments.optimization,
            trace=arguments.trace,
            progress=progress,
        )
    print(f"executable={artifact.executable_path}", flush=True)
    if arguments.frames:
        completed = artifact.run(frames=arguments.frames)
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)


if __name__ == "__main__":
    main()
