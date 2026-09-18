"""Build the managed tire standalone host from a trusted repository-SSA checkpoint.

This skips source extraction, graph planning, and frame linking.  The checkpoint
must be produced by this local compiler and pass the repository SSA self-check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.ssa_c_backend import _numpy_dtype, emit_ssa_to_c
from src.compiler.ssa_self_check import run_all
from src.compiler.vehicle_python_compilation import (
    VehiclePythonSSALowering,
    _managed_native_feeds_by_id,
    balloon_tire_managed_python_compilation_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-duration", type=float, default=1.0 / 120.0)
    parser.add_argument("--dt-initial", type=float, default=1.0 / 360.0)
    parser.add_argument("--optimization", choices=("O0",), default="O0")
    parser.add_argument(
        "--trace", action="store_true",
        help="emit authored-function SSA trace hooks",
    )
    args = parser.parse_args()

    module, outputs, exports = pickle.loads(args.checkpoint.read_bytes())
    findings = run_all(module)
    if findings:
        for finding in findings:
            print(finding, flush=True)
        raise RuntimeError(
            f"checkpoint has {len(findings)} structural finding(s)"
        )
    root_name = next(
        name for name in module.functions
        if name.endswith("__balloon_tire_managed_window")
    )
    lowered = VehiclePythonSSALowering(
        module, root_name, dict(outputs), tuple(map(str, exports)),
    )
    inputs = balloon_tire_managed_python_compilation_inputs(
        args.batch_size,
        window_duration=args.window_duration,
        dt_initial=args.dt_initial,
    )
    feeds_by_id = _managed_native_feeds_by_id(lowered, inputs.feeds)
    root = module.functions[root_name]
    arguments = {int(value.id): value for value in root.args}
    returns = [
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]
    returned = {
        int(value.id): value for values in returns for value in values
    }
    names = {
        int(value_id): name
        for name, value_id in root.metadata.get("named_outputs", ())
    }

    artifact = emit_ssa_to_c(
        module, root_name, entry_name="balloon_tire_managed_native_c",
        trace=bool(args.trace),
    )
    if not artifact.complete:
        for shortfall in artifact.shortfalls:
            print(shortfall, flush=True)
        raise RuntimeError(
            f"C emission has {len(artifact.shortfalls)} shortfall(s)"
        )
    for value_id, dtype, shape in zip(
        artifact.buffer_order,
        artifact.buffer_dtypes,
        artifact.buffer_shapes,
    ):
        value_id = int(value_id)
        if (
            value_id not in feeds_by_id
            and value_id in returned
            and value_id not in arguments
        ):
            public_shape = tuple(returned[value_id].shape or ())
            if any(
                not isinstance(extent, int) or extent < 0
                for extent in public_shape
            ):
                raise RuntimeError(
                    f"unresolved native output shape: {value_id}: "
                    f"{public_shape}"
                )
            feeds_by_id[value_id] = np.zeros(
                public_shape or (), dtype=_numpy_dtype(dtype),
            )
    missing = tuple(
        int(value_id) for value_id in artifact.buffer_order
        if int(value_id) not in feeds_by_id
    )
    if missing:
        raise RuntimeError(f"unnamed public buffers: {missing!r}")

    executable = artifact.compile_standalone(
        args.output, feeds_by_id, optimization=args.optimization,
    )
    manifest = {
        "schema": "turing.balloon-tire-managed-native.v1",
        "entrypoint": artifact.name,
        "batch_size": int(args.batch_size),
        "window_duration": float(args.window_duration),
        "dt_initial": float(args.dt_initial),
        "optimization": str(args.optimization),
        "trace": bool(args.trace),
        "checkpoint": str(args.checkpoint.resolve()),
        "buffers": [],
    }
    for index, (value_id, dtype) in enumerate(zip(
        artifact.buffer_order, artifact.buffer_dtypes,
    )):
        value_id = int(value_id)
        argument = arguments.get(value_id, returned.get(value_id))
        accounting = dict(argument.accounting or {})
        value = np.asarray(feeds_by_id[value_id])
        parameter = str(
            accounting.get("program_abi_parameter")
            or names.get(value_id)
            or value_id
        )
        field = accounting.get("program_abi_field")
        manifest["buffers"].append({
            "index": int(index),
            "value_id": value_id,
            "role": "input" if value_id in arguments else "output",
            "return_index": next((
                position
                for values in returns
                for position, result in enumerate(values)
                if int(result.id) == value_id
            ), None),
            "name": parameter if field is None else f"{parameter}.{field}",
            "parameter": parameter,
            "field": None if field is None else str(field),
            "dtype": str(dtype),
            "semantic_dtype": str(argument.dtype or ""),
            "shape": list(map(int, value.shape)),
            "element_count": int(value.size),
            "mutable": bool(accounting.get("program_abi_mutable", False)),
            "written": bool(
                accounting.get("program_abi_field_written", False)
            ),
        })
    (executable.directory / "balloon_tire_managed.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"executable={executable.executable_path}", flush=True)
    print(f"buffers={len(manifest['buffers'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
