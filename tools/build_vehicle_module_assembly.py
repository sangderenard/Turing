"""Build one vehicle module with native, Wasm, and WebGPU realizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.vehicle_python_compilation import (
    assemble_vehicle_python_module,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "vehicle_module_assembly",
    )
    parser.add_argument("--no-native", action="store_true")
    parser.add_argument(
        "--diagnostic-shell",
        action="store_true",
        help=(
            "also write a generic compiler diagnostic page; this is not the "
            "Living Data Map game surface"
        ),
    )
    arguments = parser.parse_args()

    assembly = assemble_vehicle_python_module(
        progress=lambda message: print(message, flush=True),
    )
    manifest = assembly.write(
        arguments.output,
        compile_native=not arguments.no_native,
        emit_diagnostic_shell=arguments.diagnostic_shell,
    )
    print(manifest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
