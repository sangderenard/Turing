"""An annotated scalar parameter must lower like the unannotated one.

``def bump(k: int)`` and ``def bump(k)`` are the same program.  With ``k``
declared as a scalar in ``program_abi.values``, the unannotated form passes the
full-native execution contract; the annotated form is rejected with an unnamed
formal.  Valid Python must compile, so this is a compiler defect.

    python -u tools/compiler_probes/probe_annotated_scalar_parameter.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
PLAIN = "def bump(k):\n    return k + 1\n"
ANNOTATED = "def bump(k: int):\n    return k + 1\n"
ANNOTATED_RETURN = "def bump(k: int) -> int:\n    return k + 1\n"


def lower(source: str):
    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {}, "bindings": [],
            "values": [{
                "function": "bump", "parameter": "k", "storage": "scalar",
                "dtype": "int64", "rank": 0, "python_type": "builtins.int",
            }],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            source, "bump", name="annotated_scalar", python_bindings={},
            extraction_contract=contract, runtime_closure_only=True,
        )


def main() -> int:
    failures = 0
    for label, source in (
        ("plain", PLAIN), ("annotated", ANNOTATED),
        ("annotated with return", ANNOTATED_RETURN),
    ):
        try:
            lower(source)
            print(f"ok   {label}")
        except Exception as error:  # noqa: BLE001 -- report each form
            failures += 1
            print(f"FAIL {label}: {type(error).__name__}: {str(error)[:160]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
