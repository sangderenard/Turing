"""A name-carried arm whose version is a nested in-place store.

The shape of ``_resolve_floor`` in ``engine_toy/woodshop.py``::

    def f(x, a, b):
        m = x * 1.0
        if a < 0.0:
            m[2] = m[2] * 0.5          # an in-place store: version A of m
            t = b - 1.0
            if t > 0.0:
                m[:2] -= t             # an in-place store: version B of m
        return m

Both store versions are versions of ONE storage (the store-chain rule):
``control_value_alias`` aliases each to the resident ``x * 1.0`` value, and
the control builder never binds an in-place store version.  The outer merge
of ``m`` takes version B as its true arm.  Before the fix
``_carried_name_arm`` looked B up in ``external_values`` only, found no
binding, saw B's authored ``name_binding`` row and refused it with
``carried-name-arm-missing``.

The probe lowers the program, emits C, runs it natively for every branch
combination and compares with CPython executing the same source.

    python -u tools/compiler_probes/probe_nested_inplace_arm.py
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
BUILD = REPO / "build" / "nested_inplace_arm"
NAME = "nested_inplace_arm"

SOURCE = '''
def f(x, a, b):
    m = x * 1.0
    if a < 0.0:
        m[2] = m[2] * 0.5
        t = b - 1.0
        if t > 0.0:
            m[:2] -= t
    return m
'''

X = np.array([1.5, -2.0, 3.25, 4.0])
# (a, b): outer false; outer true + inner false; outer true + inner true.
CASES = ((1.0, 3.0), (-1.0, 0.5), (-1.0, 3.0), (1.0, 0.5))


def contract() -> ExtractionContract:
    values = [{
        "function": "f", "parameter": "x", "storage": "span",
        "dtype": "float64", "rank": 1, "shape": [4], "python_type": TENSOR,
    }]
    for parameter in ("a", "b"):
        values.append({
            "function": "f", "parameter": parameter, "storage": "scalar",
            "dtype": "float64", "rank": 0, "python_type": "builtins.float",
        })
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({"records": {}, "bindings": [], "values": values})
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


def _tensor_reference():
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    return c_backend_repository_ssa_reference()


def python_result(a: float, b: float) -> np.ndarray:
    namespace: dict = {}
    exec(SOURCE, namespace)  # noqa: S102 -- the probe's own source
    return np.asarray(namespace["f"](X.copy(), a, b))


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            module, outputs, _ = lower_ast_source_to_ssa(
                SOURCE, "f", name=NAME, python_bindings={},
                extraction_contract=contract(), runtime_closure_only=True,
                tensor_ssa_reference=_tensor_reference(),
            )
        except Exception as error:  # noqa: BLE001 -- the defect surfaces here
            print(f"FAIL lowering raised {type(error).__name__}: {error}")
            return 1
    print("ok   the program lowers")
    root = module.functions[f"{NAME}__f"]
    named = dict(root.metadata.get("parameter_names") or ())
    artifact = emit_ssa_module_to_c(module, root.name)
    if not artifact.complete:
        print(f"FAIL C emission incomplete: {artifact.shortfalls}")
        return 1
    BUILD.mkdir(parents=True, exist_ok=True)
    artifact.compile(BUILD / NAME)
    failures = 0
    for a, b in CASES:
        expected = python_result(a, b)
        feeds = {
            int(named["x"]): X.copy(),
            int(named["a"]): np.asarray(a, dtype=np.float64),
            int(named["b"]): np.asarray(b, dtype=np.float64),
        }
        result = artifact.prepare_execution(feeds).run()
        actual = np.asarray(result.buffers[outputs[root.name][0].id])
        equal = (
            actual.size == expected.size
            and np.array_equal(actual.reshape(-1), expected.reshape(-1))
        )
        failures += 0 if equal else 1
        print(f"{'ok  ' if equal else 'FAIL'} a={a} b={b} "
              f"python={expected.tolist()} native={actual.tolist()}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
