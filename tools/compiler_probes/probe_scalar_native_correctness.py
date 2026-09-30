"""Scalar programs run natively and equal Python, annotated or not.

A parameter annotated ``bool/float/int/str`` or given a scalar default is
marked scalar by the reducer.  Until 2026-09-30 the deployment classifier then
deferred every all-scalar expression to its consumer, and a returned scalar (or
a scalar feeding a tensor op) had no producer (see
``docs/DECISION_scalar_expression_deferral_2026-09-30.md``).  Each program
below is lowered, emitted as C, compiled, run, and compared with CPython
executing the same source.  Every program appears with and without the
annotation: they are one program, so they must agree with Python and with each
other.

    python -u tools/compiler_probes/probe_scalar_native_correctness.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ssa_c_backend import emit_ssa_module_to_c  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
TENSOR = "src.common.tensors.abstraction.AbstractTensor"

BUILD = REPO / "build" / "scalar_native_correctness"

X = np.array([1.5, -2.0, 3.25, 4.0])

# name -> (source template, declared scalar dtype, scalar value, has tensor)
PROGRAMS = {
    "bump": ("def f(k{a}):\n    return k + 1\n", "int64", 6, False),
    "chain": ("def f(k{a}):\n    return (k + 1) * 2\n", "int64", 6, False),
    "twice": ("def f(k{a}):\n    t = k + 1\n    return t * t\n", "int64", 6, False),
    "cond": ("def f(k{a}):\n    if k + 1 > 3:\n        return k\n    return 0\n",
             "int64", 6, False),
    "loop": ("def f(k{a}):\n    s = 0\n    for i in range(k + 1):\n"
             "        s = s + i\n    return s\n", "int64", 6, False),
    "scale": ("def f(x, dt{a}):\n    return x * (dt * 0.5)\n", "float64", 0.375, True),
    "shared": ("def f(x, dt{a}):\n    h = dt * 0.5\n    return x * h + x * h\n",
               "float64", 0.375, True),
}


def contract(dtype: str, has_tensor: bool) -> ExtractionContract:
    values = []
    if has_tensor:
        values.append({
            "function": "f", "parameter": "x", "storage": "span",
            "dtype": "float64", "rank": 1, "shape": [4], "python_type": TENSOR,
        })
    values.append({
        "function": "f", "parameter": "dt" if has_tensor else "k",
        "storage": "scalar", "dtype": dtype, "rank": 0,
        "python_type": "builtins.float" if dtype == "float64" else "builtins.int",
    })
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({"records": {}, "bindings": [], "values": values})
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


def python_result(source: str, has_tensor: bool, scalar):
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 -- the probe's own source
    if has_tensor:
        return np.asarray(namespace["f"](X.copy(), scalar))
    return np.asarray(namespace["f"](scalar))


def native_result(source: str, dtype: str, has_tensor: bool, scalar, workdir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _ = lower_ast_source_to_ssa(
            source, "f", name="scalar_native", python_bindings={},
            extraction_contract=contract(dtype, has_tensor),
            runtime_closure_only=True,
            **({"tensor_ssa_reference": _tensor_reference()} if has_tensor else {}),
        )
    root = module.functions["scalar_native__f"]
    named = dict(root.metadata.get("parameter_names") or ())
    artifact = emit_ssa_module_to_c(module, root.name)
    if not artifact.complete:
        raise RuntimeError(f"C emission incomplete: {artifact.shortfalls}")
    artifact.compile(workdir / "scalar_native")
    scalar_name = "dt" if has_tensor else "k"
    feeds = {
        int(named[scalar_name]): np.asarray(
            scalar, dtype=np.float64 if dtype == "float64" else np.int64
        ),
    }
    if has_tensor:
        feeds[int(named["x"])] = X.copy()
    result = artifact.prepare_execution(feeds).run()
    return np.asarray(result.buffers[outputs[root.name][0].id])


def _tensor_reference():
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    return c_backend_repository_ssa_reference()


def main() -> int:
    failures = 0
    for name, (template, dtype, scalar, has_tensor) in PROGRAMS.items():
        expected = python_result(template.format(a=""), has_tensor, scalar)
        for annotation in ("", ": float" if dtype == "float64" else ": int"):
            label = f"{name:7} {'annotated' if annotation else 'plain    '}"
            source = template.format(a=annotation)
            try:
                # The compiled library stays loaded in this process, so its
                # directory cannot be removed on Windows; build/ is ignored.
                workdir = BUILD / label.replace(" ", "_")
                workdir.mkdir(parents=True, exist_ok=True)
                actual = native_result(source, dtype, has_tensor, scalar, workdir)
                # A scalar result crosses the native boundary as a one-cell
                # buffer, so compare the values, not the shapes.
                equal = (
                    actual.size == expected.size
                    and np.array_equal(actual.reshape(-1), expected.reshape(-1))
                )
                same_kind = actual.dtype.kind == expected.dtype.kind
                note = "" if same_kind else (
                    f"  [dtype: python {expected.dtype}, native {actual.dtype}]"
                )
                print(f"{'ok  ' if equal else 'FAIL'} {label} "
                      f"python={expected.tolist()} native={actual.tolist()}{note}")
                failures += 0 if equal else 1
            except Exception as error:  # noqa: BLE001 -- report each program
                failures += 1
                print(f"FAIL {label} {type(error).__name__}: {str(error)[:140]}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
