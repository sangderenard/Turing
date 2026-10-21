"""Every contracted value owns a ``value_type_abi`` row from its contract.

Program: a root taking a plain ``float`` parameter, a span parameter and a
record parameter read as ``metrics.max_vel`` (``class Metrics: max_vel:
float``).  Wrong record: the ProgramABI contract (storage, rank, dtype,
shape) lived only in ``program_abi`` dicts and accounting; ``value_type_abi``
was posted only when the contract named a ``python_type``, so ``_contract_
physical`` found no row for a plain scalar parameter or a primitive record
field (audit ``ssa-physical-unposted``).  Rule: the contract is the
declaration -- the value's class is read from the row it was ingested from
(``parameter_annotation`` / ``class_field_declaration``) and the one
VALUE_TYPE_ABI post carries storage/rank/dtype/shape.
"""
from __future__ import annotations

from pathlib import Path

from src.common.dt_system.dt_scaler import Metrics
from src.common.tensors.abstraction import AbstractTensor
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, VALUE_TYPE_ABI,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.precompile_to_ssa import _contract_physical

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

SOURCE = """
def tick(metrics: Metrics, x: float, v: list[float]):
    return metrics.max_vel + x + v[0]
"""


def _contract():
    return ExtractionContract(CONTRACTS / "program_extraction.yaml")


def test_every_contracted_value_has_a_value_type_abi_row():
    module, _, _ = lower_ast_source_to_ssa(
        SOURCE, "tick", name="contract_rows",
        python_bindings={"AbstractTensor": AbstractTensor, "Metrics": Metrics},
        runtime_closure_only=True, extraction_contract=_contract(),
        progress=lambda message: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    root = next(
        fn for name, fn in module.functions.items() if name.endswith("tick")
    )
    page = book.page(VALUE_TYPE_ABI)
    rows = {int(row[1]): row for row in page.rows()}
    formals = {
        int(formal.id): formal for formal in root.args
        if (formal.accounting or {}).get("program_abi_storage") is not None
        # Compiler-minted descriptor members of a span (int64 length /
        # capacity words) are not authored contract values.
        and int(formal.id) < (1 << 40)
    }
    field = next(
        formal for formal in formals.values()
        if formal.accounting.get("program_abi_field") == "max_vel"
    )
    plain = [
        formal for formal in formals.values()
        if formal.accounting.get("program_abi_field") is None
    ]
    scalar = next(f for f in plain if f.accounting["program_abi_storage"] == "scalar")
    span = next(f for f in plain if f.accounting["program_abi_storage"] == "span")
    expected = {
        int(scalar.id): ("scalar", "float64", 0),
        int(span.id): ("span", "float64", 1),
        int(field.id): ("scalar", "float64", 0),
    }
    for value_id, view in expected.items():
        assert value_id in rows, f"value {value_id} has no value_type_abi row"
        arms = page.latest(rows[value_id]).physical_alternatives
        assert len(arms) == 1
        assert (arms[0].storage, arms[0].dtype, arms[0].rank) == view
    # The record parameter owns its own row (record storage).
    assert any(
        arm.storage == "record"
        for row in rows.values()
        for arm in page.latest(row).physical_alternatives
    )
    # The field formal's physical view is derived from the row: the helper
    # precompile uses for a root/region formal returns storage and rank.
    cell = book.latest_ref(CANONICAL_VALUE, rows[int(field.id)])
    derived = _contract_physical(book, field, cell)
    assert derived is not None
    assert derived[0] == {"storage": "scalar", "rank": 0}
    # Accounting mirrors the posted fact for every contracted formal.
    for value_id, view in expected.items():
        accounting = formals[value_id].accounting
        assert (
            accounting["program_abi_storage"], accounting["program_abi_rank"],
        ) == (view[0], view[2])
    findings = CorrelationTable.build(module).findings(module)
    assert not [f for f in findings if f.kind == "ssa-physical-unposted"]
    assert not [f for f in findings if "unsourced" in f.kind], findings
