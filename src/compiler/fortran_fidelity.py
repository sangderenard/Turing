"""Compile emitted Fortran and record numerical parity with its source IR.

This is deliberately a verification boundary, not another lowering path.  It
executes the exact :class:`FortranModule` whose source is published by a site
bundle and compares its observable outputs with the NumPy rendering of the
same ``FusedProgram``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


FORTRAN_FIDELITY_SCHEMA = "turing-fortran-fidelity-v1"

_CTYPES = {
    "bool": ctypes.c_bool,
    "logical": ctypes.c_bool,
    "float": ctypes.c_float,
    "float32": ctypes.c_float,
    "f32": ctypes.c_float,
    "double": ctypes.c_double,
    "float64": ctypes.c_double,
    "f64": ctypes.c_double,
    "int": ctypes.c_int32,
    "int32": ctypes.c_int32,
    "i32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "i64": ctypes.c_int64,
}

_NUMPY_DTYPES = {
    "bool": "bool",
    "logical": "bool",
    "float": "float32",
    "float32": "float32",
    "f32": "float32",
    "double": "float64",
    "float64": "float64",
    "f64": "float64",
    "int": "int32",
    "int32": "int32",
    "i32": "int32",
    "int64": "int64",
    "i64": "int64",
}


def deterministic_verification_cases(
    feeds: Mapping[int, Any],
) -> tuple[tuple[str, dict[int, Any]], ...]:
    """Make three deterministic, shape-preserving cases from captured feeds."""

    import numpy as np

    base = {int(key): np.asarray(value).copy() for key, value in feeds.items()}
    reversed_case = {
        key: (np.flip(value).copy() if value.ndim else value.copy())
        for key, value in base.items()
    }
    swept: dict[int, Any] = {}
    for feed_index, (key, value) in enumerate(sorted(base.items())):
        if not value.shape:
            swept[key] = value.copy()
        elif value.dtype.kind == "b":
            swept[key] = (np.arange(value.size) % 2 == feed_index % 2).reshape(value.shape)
        elif value.dtype.kind in "iu":
            swept[key] = np.arange(value.size, dtype=value.dtype).reshape(value.shape)
        else:
            swept[key] = np.linspace(
                -0.75 + 0.125 * feed_index,
                1.25 + 0.125 * feed_index,
                value.size,
                dtype=value.dtype,
            ).reshape(value.shape)
    return (
        ("captured-probe", base),
        ("reversed-probe", reversed_case),
        ("deterministic-sweep", swept),
    )


def _compiler_identity(compiler: str) -> dict[str, str]:
    completed = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    )
    first_line = (completed.stdout or completed.stderr).splitlines()
    return {
        "path": str(Path(compiler).resolve()),
        "version": first_line[0].strip() if first_line else "unknown",
    }


def _ordered(array: Any, dtype: str):
    import numpy as np

    result = np.asarray(array, dtype=_NUMPY_DTYPES.get(dtype, "float64"))
    return np.asfortranarray(result) if result.ndim else result


def _json_value(array: Any) -> Any:
    value = array.tolist()
    return value


def verify_fortran_module(
    module: Any,
    program: Any,
    feeds: Mapping[int, Any],
    directory: str | Path,
    *,
    entrypoint: str,
    cases: Sequence[tuple[str, Mapping[int, Any]]] | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compile ``module``, run cases, and return a self-contained proof record.

    A mismatch raises ``AssertionError`` so a bundle can never claim successful
    verification while publishing numerically different Fortran.
    """

    import numpy as np

    from ..common.tensors.fused_ir import ordered_feed_ids
    from .fused_program_python_backend import compile_single_region_python
    from .ssa_fortran_backend import compile_module, fortran_compiler

    compiler = fortran_compiler()
    if compiler is None:
        raise RuntimeError("a Fortran compiler is required for fidelity verification")
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    library_path = compile_module(module, directory=output_directory)

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        # Keep the handle alive until all native calls have completed.
        dll_directory = os.add_dll_directory(str(Path(compiler).parent))
    else:
        dll_directory = None
    library = None
    try:
        library = ctypes.CDLL(str(library_path))
        native = getattr(library, module.api.entry_point(entrypoint).symbol)
        api = module.api.entry_point(entrypoint)
        feed_ids = ordered_feed_ids(program)
        feed_origins = dict(
            (getattr(program, "extras", None) or {}).get(
                "capture_feed_origins", {}
            )
        )
        feed_names = {feed_id: f"feed{index}" for index, feed_id in enumerate(feed_ids)}
        reference = compile_single_region_python(
            program,
            feed_names,
            dialect="numpy",
            function_name=f"{entrypoint}_reference",
        ).callable
        selected_cases = tuple(cases or deterministic_verification_cases(feeds))
        case_records: list[dict[str, Any]] = []

        for case_name, case_feeds in selected_cases:
            missing = set(feed_ids) - set(case_feeds)
            if missing:
                raise ValueError(f"{case_name} is missing feed ids {sorted(missing)}")
            inputs = {
                feed_id: np.asarray(case_feeds[feed_id]) for feed_id in feed_ids
            }
            expected_raw = reference(*(inputs[feed_id] for feed_id in feed_ids))
            expected_values = (
                tuple(expected_raw)
                if len(program.outputs) > 1
                else (expected_raw,)
            )
            expected_by_id = {
                value_id: np.asarray(value)
                for value_id, value in zip(program.outputs.values(), expected_values)
            }
            native_arrays: dict[int, Any] = {}
            call_arguments: list[Any] = []
            argument_types: list[Any] = []
            for parameter in api.parameters:
                dtype_name = str(parameter.dtype).casefold()
                ctype = _CTYPES.get(dtype_name, ctypes.c_double)
                if parameter.role == "extent":
                    extent = int(parameter.name.rsplit("_", 1)[-1])
                    call_arguments.append(ctype(extent))
                    argument_types.append(ctype)
                    continue
                value_id = int(parameter.name[1:])
                if parameter.role in {"input", "inout"}:
                    array = _ordered(inputs[value_id], dtype_name)
                    if parameter.role == "inout":
                        native_arrays[value_id] = array
                else:
                    shape = tuple(parameter.shape)
                    array = _ordered(np.empty(shape or (), dtype=_NUMPY_DTYPES.get(
                        dtype_name, "float64"
                    )), dtype_name)
                    native_arrays[value_id] = array
                if parameter.passing == "value":
                    call_arguments.append(ctype(array.item()))
                    argument_types.append(ctype)
                else:
                    pointer = ctypes.POINTER(ctype)
                    call_arguments.append(array.ctypes.data_as(pointer))
                    argument_types.append(pointer)
            native.argtypes = argument_types
            native.restype = None
            native(*call_arguments)

            outputs = []
            case_passed = True
            for output_name, value_id in program.outputs.items():
                expected = expected_by_id[value_id]
                actual = np.asarray(native_arrays[value_id])
                passed = bool(np.allclose(
                    actual, expected, rtol=rtol, atol=atol, equal_nan=True
                ))
                case_passed = case_passed and passed
                finite = np.isfinite(actual) & np.isfinite(expected)
                absolute_error = (
                    float(np.max(np.abs(actual[finite] - expected[finite])))
                    if np.any(finite) else 0.0
                )
                outputs.append({
                    "name": output_name,
                    "value_id": int(value_id),
                    "shape": list(actual.shape),
                    "reference": _json_value(expected),
                    "fortran": _json_value(actual),
                    "max_absolute_error": absolute_error,
                    "passed": passed,
                })
            record = {
                "name": case_name,
                "inputs": [
                    {
                        "name": feed_origins.get(
                            feed_id, feed_origins.get(str(feed_id), {})
                        ).get("binding_name", f"feed_{feed_id}"),
                        "value_id": int(feed_id),
                        "value": _json_value(inputs[feed_id]),
                    }
                    for feed_id in feed_ids
                ],
                "outputs": outputs,
                "passed": case_passed,
            }
            case_records.append(record)
            if not case_passed:
                raise AssertionError(
                    f"generated Fortran disagrees with the reference in {case_name}: "
                    + json.dumps(outputs, separators=(",", ":"))
                )
    finally:
        if library is not None:
            # Windows locks a loaded DLL.  Release it explicitly so temporary
            # verification bundles remain removable and atomic publication
            # can rename their directories without a lingering native handle.
            import _ctypes

            if os.name == "nt":
                _ctypes.FreeLibrary(library._handle)
            else:  # pragma: no cover - exercised on POSIX builders
                _ctypes.dlclose(library._handle)
        if dll_directory is not None:
            dll_directory.close()

    source_bytes = module.source.encode("utf-8")
    return {
        "schema": FORTRAN_FIDELITY_SCHEMA,
        "passed": all(case["passed"] for case in case_records),
        "entrypoint": entrypoint,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "native_library": library_path.name,
        "compiler": _compiler_identity(compiler),
        "tolerances": {"relative": rtol, "absolute": atol},
        "case_count": len(case_records),
        "cases": case_records,
    }


__all__ = [
    "FORTRAN_FIDELITY_SCHEMA",
    "deterministic_verification_cases",
    "verify_fortran_module",
]
