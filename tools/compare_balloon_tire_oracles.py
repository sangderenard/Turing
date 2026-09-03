"""Compare authored NumPy, generated C, and retained native balloon trajectories.

The retained native appendage is a diagnostic oracle, not product authority.
All three paths receive the same canonical input row and initialized material
state.  The tool reports the first cross-frame disagreement without changing
the authored equations, fixture, timestep, or acceptance policy.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.compiler.vehicle_balloon_tire import (
    compile_balloon_bead_implicit_step_ssa,
    compile_balloon_gas_ssa,
    compile_balloon_membrane_face_ssa,
)
from src.compiler.vehicle_balloon_tire_native import (
    compile_native_balloon_tire_assembly,
)
from src.compiler.vehicle_python_compilation import (
    balloon_tire_default_initialized_state,
    balloon_tire_python_compilation_inputs,
)


def _numpy_link(compilation):
    argument_names = tuple(compilation.function.metadata["argument_names"])
    output_names = tuple(compilation.function.metadata["output_names"])
    arguments = tuple(sp.Symbol(name) for name in argument_names)
    expressions = tuple(equation.rhs for equation in compilation.equations)
    evaluate = sp.lambdify(arguments, expressions, modules="numpy")

    def linked(*values):
        arrays = tuple(
            np.asarray(value.data if hasattr(value, "data") else value)
            for value in values
        )
        broadcast = np.broadcast_arrays(*arrays)
        result = evaluate(*broadcast)
        if len(output_names) == 1:
            result = (result,)
        shape = broadcast[0].shape
        return tuple(
            NumPyTensorOperations.tensor(
                np.array(np.broadcast_to(value, shape), dtype=np.float64, copy=True)
            )
            for value in result
        )

    return linked


def _eager_snapshots(checkpoints: tuple[int, ...], feed_overrides=None):
    inputs = balloon_tire_python_compilation_inputs()
    # Preserve semantic index/mask dtypes for eager execution.  The generated
    # native extraction contract stores those same public buffers physically
    # as doubles and performs its explicit gather/mask conversions internally.
    feeds = {
        name: np.asarray(value).copy()
        for name, value in inputs.feeds.items()
    }
    feeds["state"] = balloon_tire_default_initialized_state(
        feeds["inputs"], feeds["wheel_input_indices"], feeds["rest"]
    )
    if feed_overrides:
        for name, value in feed_overrides.items():
            feeds[name] = np.asarray(value).copy()
    namespace = {
        "AbstractTensor": NumPyTensorOperations,
        "balloon_tire_gas": _numpy_link(compile_balloon_gas_ssa()),
        "balloon_tire_membrane_face": _numpy_link(
            compile_balloon_membrane_face_ssa()
        ),
        "balloon_tire_bead_implicit_step": _numpy_link(
            compile_balloon_bead_implicit_step_ssa()
        ),
    }
    exec(inputs.source, namespace)
    step = namespace[inputs.entrypoint]
    tensors = {
        name: NumPyTensorOperations.tensor(value)
        for name, value in feeds.items()
    }
    state, output = tensors["state"], tensors["output"]
    snapshots = {}
    for frame in range(1, max(checkpoints) + 1):
        # Instability is part of the verdict and is summarized compactly
        # below; NumPy's expanded lambdified warnings can otherwise be
        # hundreds of kilobytes per frame.
        with np.errstate(all="ignore"):
            state, output = step(
                tensors["inputs"], state, output,
                tensors["wheel_input_indices"], tensors["rest"],
                tensors["face_vertices"], tensors["face_rest"],
                tensors["face_scatter"], tensors["bending_incidence"],
                tensors["bending_scatter"], tensors["bending_weight"],
                tensors["vertex_area"], tensors["bead_mask"],
            )
        if frame in checkpoints:
            snapshots[frame] = (
                np.asarray(state.data).copy(), np.asarray(output.data).copy()
            )
    return feeds, snapshots


def _generated_snapshots(executable: Path, feeds, checkpoints: tuple[int, ...]):
    counts = [np.asarray(value).size for value in feeds.values()]
    state_offset = counts[0]
    output_offset = state_offset + counts[1]
    snapshots = {}
    for frame in checkpoints:
        subprocess.run(
            [str(executable), str(frame)], check=True,
            cwd=executable.parent, stdout=subprocess.DEVNULL,
        )
        flat = np.fromfile(executable.parent / "final-outputs.bin", dtype="<f8")
        state = flat[state_offset:output_offset].reshape(feeds["state"].shape)
        output = flat[output_offset:output_offset + counts[2]].reshape(
            feeds["output"].shape
        )
        snapshots[frame] = (state.copy(), output.copy())
    return snapshots


def _retained_snapshots(library_path: Path, feeds, checkpoints: tuple[int, ...]):
    assembly = compile_native_balloon_tire_assembly()
    inputs = balloon_tire_python_compilation_inputs()
    canonical_names = tuple(inputs.feeds.keys())
    if canonical_names[:3] != ("inputs", "state", "output"):
        raise RuntimeError(f"unexpected canonical buffer order: {canonical_names!r}")
    input_row = feeds["inputs"][0]
    if input_row.size != len(assembly.input_names):
        raise RuntimeError(
            f"retained input ABI mismatch: {input_row.size} != {len(assembly.input_names)}"
        )
    library = ctypes.CDLL(str(library_path))
    input_buffer = (ctypes.c_double * input_row.size)(*map(float, input_row))
    state_buffer = (ctypes.c_double * assembly.state_scalar_count)()
    output_buffer = (ctypes.c_double * len(assembly.output_names))()
    library.balloon_tire_appendage_initialize.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    library.balloon_tire_appendage_step.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.balloon_tire_appendage_initialize(input_buffer, state_buffer)
    snapshots = {}
    for frame in range(1, max(checkpoints) + 1):
        library.balloon_tire_appendage_step(
            input_buffer, state_buffer, output_buffer
        )
        if frame in checkpoints:
            snapshots[frame] = (
                np.ctypeslib.as_array(state_buffer).copy().reshape(4, -1, 6),
                np.ctypeslib.as_array(output_buffer).copy().reshape(4, 14),
            )
    return snapshots


def _max_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.all():
        return float("inf")
    return float(np.max(np.abs(left - right)))


def _describe(name: str, value: np.ndarray) -> str:
    finite = np.isfinite(value)
    maximum = float(np.max(np.abs(value[finite]))) if finite.any() else float("nan")
    return f"{name}_finite={bool(finite.all())} {name}_max_abs={maximum:.9g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-exe", type=Path, required=True)
    parser.add_argument("--retained-library", type=Path, required=True)
    parser.add_argument(
        "--frames", type=int, nargs="+", default=(1, 2, 3, 4, 5, 6, 8, 10, 60)
    )
    args = parser.parse_args()
    checkpoints = tuple(sorted(set(args.frames)))
    if not checkpoints or checkpoints[0] < 1:
        parser.error("--frames must contain positive frame numbers")

    feeds, eager = _eager_snapshots(checkpoints)
    generated = _generated_snapshots(
        args.generated_exe.resolve(), feeds, checkpoints
    )
    retained = _retained_snapshots(
        args.retained_library.resolve(), feeds, checkpoints
    )
    print("frame path state/output status and lane-0 maximum absolute error")
    for frame in checkpoints:
        eager_state, eager_output = eager[frame]
        generated_state, generated_output = generated[frame]
        retained_state, retained_output = retained[frame]
        eager_lane = (eager_state[0], eager_output[0])
        generated_lane = (generated_state[0], generated_output[0])
        print(
            f"frame={frame} "
            f"{_describe('numpy_state', eager_state)} "
            f"{_describe('generated_state', generated_state)} "
            f"{_describe('retained_state', retained_state)}"
        )
        print(
            "  errors "
            f"generated_numpy_state={_max_error(generated_lane[0], eager_lane[0]):.9g} "
            f"generated_numpy_output={_max_error(generated_lane[1], eager_lane[1]):.9g} "
            f"retained_numpy_state={_max_error(retained_state, eager_lane[0]):.9g} "
            f"retained_numpy_output={_max_error(retained_output, eager_lane[1]):.9g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
