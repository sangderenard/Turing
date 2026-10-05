"""An ``if`` arm whose only statements are element writes into span fields.

Modelled on ``probe_scalar_write_only_arm.py`` (same record contract shape,
same ``program_abi`` binding, same lowering entry) but the arm writes one
ELEMENT of two rank-1 span fields instead of a whole scalar field, exactly
as ``dt_controller.step_with_dt_control_used`` does on its retry loop::

    if rejected:
        metrics.control_values[0] = float(dt_for_advance)
        metrics.control_present[0] = 1.0

The reducer posts an ELEMENT_WRITTEN ``reducer_field_state`` cell for each
store (``obj.field[i] = v`` mutates ``obj.field`` in place) and merges the
field at the ``if`` with ``field_state_arms = (element cell, pre-branch
cell, test cell)``.  The control builder's ``_carried_field_arm`` then
reads the arm's ``ssa_field_version`` at the ELEMENT_WRITTEN cell.

    python -u tools/compiler_probes/probe_element_written_field_arm.py

Green when the program lowers with no ``carried-field-arm-missing``
shortfall, every ELEMENT_WRITTEN cell of the two fields has an
``ssa_field_version`` row whose edges name the cell, the store's
``control_value_alias`` PLANNING row and the storage's binding row, and
each MERGED cell's version derives from its ELEMENT_WRITTEN version.

Before the fix (``_ControlSSABuilder._carried_field_arm`` knew no writer for
an ELEMENT_WRITTEN cell) the lowering raised::

    carried-field-arm-missing (a field-carried conditional arm names a
    field-state cell with no ssa_field_version: row=(..., Ref(
    'reducer_field_state', (..., 'control_present'), 1)))

the same shortfall the N=2 orbital dt system reported on
``step_with_dt_control_used``.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CONTROL_VALUE_ALIAS,
    CONTROL_VALUE_BINDING,
    REDUCER_FIELD_STATE,
    SSA_FIELD_VERSION,
    FieldStateKind,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import Ref, identity_book  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
NAME = "element_written_field_arm"
FIELDS = ("control_values", "control_present")
PREDICATE_PARAMETER = "rejected"

SOURCE = '''
from dataclasses import dataclass, field

from src.common.tensors.abstraction import AbstractTensor


@dataclass
class Metrics:
    hard_failure: bool = False
    control_values: AbstractTensor = field(
        default_factory=lambda: AbstractTensor.zeros(10, dtype=float)
    )
    control_present: AbstractTensor = field(
        default_factory=lambda: AbstractTensor.zeros(10, dtype=float)
    )


def step(m: Metrics, rejected: bool, dt: float) -> Metrics:
    if bool(m.hard_failure):
        return m
    retries = 0
    while retries < 3:
        retries += 1
        if rejected:
            m.control_values[0] = float(dt)
            m.control_present[0] = 1.0
    return m
'''


def contract():
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Metrics": {
                "identity": f"{NAME}.Metrics",
                "fields": {
                    "hard_failure": {"storage": "scalar", "dtype": "bool",
                                     "mutable": True, "default": False},
                    "control_values": {
                        "storage": "span", "dtype": "float64", "rank": 1,
                        "shape": [10], "mutable": True, "default": 0.0,
                    },
                    "control_present": {
                        "storage": "span", "dtype": "float64", "rank": 1,
                        "shape": [10], "mutable": True, "default": 0.0,
                    },
                },
            }},
            "bindings": [
                {"function": "step", "parameter": "m", "record": "Metrics"},
            ],
            "values": [
                {"function": "step", "parameter": PREDICATE_PARAMETER,
                 "storage": "scalar", "dtype": "bool", "rank": 0,
                 "python_type": "builtins.bool"},
                {"function": "step", "parameter": "dt",
                 "storage": "scalar", "dtype": "float64", "rank": 0,
                 "python_type": "builtins.float"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


def lower():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            SOURCE, "step", name=NAME, python_bindings={},
            extraction_contract=contract(), runtime_closure_only=True,
        )


failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def main() -> int:
    try:
        module, _outputs, _exports = lower()
    except Exception as exc:  # noqa: BLE001 -- the defect surfaces here today
        print(f"lowering raised {type(exc).__name__}: {exc}")
        check("the program lowers", False)
        return 1
    check("the program lowers", True)
    shortfalls = tuple((module.metadata or {}).get("lowering_shortfalls", ()))
    for line in shortfalls:
        print(f"  shortfall: {line}")
    check("no carried-field-arm-missing shortfall", not any(
        "carried-field-arm-missing" in str(line) for line in shortfalls
    ))
    book = identity_book(module)

    def kind_at(cell):
        if not isinstance(cell, Ref):
            return None
        page = book.pages.get(cell.page.name)
        fact = None if page is None else page.cells.get((cell.row, cell.column))
        return getattr(fact, "kind", None)

    versions = book.pages.get(SSA_FIELD_VERSION.name)
    for field in FIELDS:
        element_rows = []
        merged_rows = []
        for row in versions.rows() if versions is not None else ():
            cell = row[-1]
            if not (
                isinstance(cell, Ref)
                and cell.page.name == REDUCER_FIELD_STATE.name
                and cell.row[-1] == field
            ):
                continue
            kind = kind_at(cell)
            if kind is FieldStateKind.ELEMENT_WRITTEN:
                element_rows.append(row)
            elif kind is FieldStateKind.MERGED:
                merged_rows.append(row)
        print(f"{field}: ssa_field_version rows at ELEMENT_WRITTEN cells: "
              f"{len(element_rows)}, at MERGED cells: {len(merged_rows)}")
        check(f"{field}: a version is posted at its ELEMENT_WRITTEN cell",
              bool(element_rows))
        check(f"{field}: a version is posted at its MERGED cell",
              bool(merged_rows))
        element_refs = []
        for row in element_rows:
            ref = book.latest_ref(SSA_FIELD_VERSION, row)
            element_refs.append(ref)
            sources = tuple(source for source, _stage in book.edges_into(ref))
            for source in sources:
                print(f"    {ref!r}\n        <- {source!r}")
            pages = {source.page.name for source in sources}
            check(
                f"{field}: the ELEMENT_WRITTEN version derives from the cell, "
                "the store's PLANNING alias and the storage's binding",
                {REDUCER_FIELD_STATE.name, CONTROL_VALUE_ALIAS.name,
                 CONTROL_VALUE_BINDING.name} <= pages,
            )
        for row in merged_rows:
            ref = book.latest_ref(SSA_FIELD_VERSION, row)
            sources = tuple(source for source, _stage in book.edges_into(ref))
            check(
                f"{field}: the MERGED version derives from an ELEMENT_WRITTEN "
                "version cell",
                any(source in element_refs for source in sources),
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
