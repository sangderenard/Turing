"""Compile and RUN symbolic_fluid_frame natively from a captured pickle.

The campaign's standard of proof: a clean shortfall census is not evidence.
Usage: python tools/run_frame_native.py [build_dir] [grid]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.ssa_llvm_backend import (  # noqa: E402
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.compiler.string_table import string_token  # noqa: E402


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "build/sfdc-keyed-get10")
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with (directory / "control_repository_ssa.pkl").open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    name = "symbolic_fluid_control__symbolic_fluid_frame"
    artifact = emit_ssa_function_to_llvm(module, name)
    assert artifact.shortfalls == (), artifact.shortfalls[:3]
    native = compile_artifact(artifact, directory=directory / "frame-native")
    print(f"compiled: {len(native.buffer_order)} buffers")
    function = module.functions[name]
    feeds = {}
    scalar_feeds = {
        "dx": 1.0, "gravity": 9.81, "cfl": 0.45, "div_max": 1.0,
        "mass_max": 1.0, "minimum_height": 1e-6, "dt_max": 1.0 / 30.0,
        "dt_min": 1e-8, "Kp": 0.4, "Ki": 0.05, "A": 1.5, "shrink": 0.5,
    }
    for argument in function.args:
        accounting = dict(argument.accounting or {})
        field = str(accounting.get("program_abi_field") or "")
        value_id = int(argument.id)
        if accounting.get("program_abi_rank") == 2:
            feeds[value_id] = (
                np.ones((grid, grid)) if field == "height"
                else np.zeros((grid, grid))
            )
        elif field.endswith(".keys"):
            feeds[value_id] = np.array([
                string_token("height_positivity"),
                string_token("tracer_bounds"),
            ], dtype=np.int64)
        elif field.endswith(".values"):
            feeds[value_id] = np.zeros(2, dtype=np.float64)
        elif field.endswith(".length"):
            feeds[value_id] = np.array([2], dtype=np.int64)
        elif field in scalar_feeds:
            feeds[value_id] = scalar_feeds[field]
    execution = prepare_artifact_execution(native, feeds)
    execution.run()
    print("RAN")
    for argument in function.args:
        field = (argument.accounting or {}).get("program_abi_field")
        if field in {"height", "last_wave_speed", "last_height_violation"}:
            stored = execution.buffers.get(int(argument.id))
            if stored is None:
                print(f"  {field}: (no public buffer under id {argument.id})")
                continue
            value = np.asarray(stored)
            print(f"  {field}: min={value.min():.6g} max={value.max():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
