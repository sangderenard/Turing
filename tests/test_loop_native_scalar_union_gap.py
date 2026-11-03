"""A possibly empty loop cannot transport distinct scalar arms in one Phi."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import (
    LOOP_NATIVE_SCALAR_UNION_GAP, VALUE_TYPE_ABI,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, IdentityBook


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)
SOURCE = '''
def root(xs: list[float]):
    total = 0
    for token in xs:
        total += token
    return total
'''


def test_loop_scalar_union_names_missing_header_exit_projection(monkeypatch):
    captured = []
    original = IdentityBook.post

    def capture(self, page, row, fact, *args, **kwargs):
        result = original(self, page, row, fact, *args, **kwargs)
        if page == LOOP_NATIVE_SCALAR_UNION_GAP:
            captured.append((self, result))
        return result

    monkeypatch.setattr(IdentityBook, "post", capture)
    with pytest.raises(
        Exception,
        match=r"native_loop_scalar_union_projection_missing: loop=4, initial=2, updated=3",
    ):
        lower_ast_source_to_ssa(
            SOURCE, "root", name="loop_native_scalar_union_gap",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None,
        )
    assert len(captured) == 1
    book, gap = captured[0]
    fact = book.page(LOOP_NATIVE_SCALAR_UNION_GAP).cell(
        gap.row, gap.column)
    assert fact.operation == "header_exit_scalar_arm_projection"
    assert fact.binding_cell.row == gap.row
    assert fact.loop_cell.row[1] == 4
    assert (fact.initial_abi_cell.page, fact.updated_abi_cell.page) == (
        VALUE_TYPE_ABI, VALUE_TYPE_ABI,
    )
    initial = book.page(VALUE_TYPE_ABI).cell(
        fact.initial_abi_cell.row, fact.initial_abi_cell.column)
    updated = book.page(VALUE_TYPE_ABI).cell(
        fact.updated_abi_cell.row, fact.updated_abi_cell.column)
    assert (initial.alternatives, updated.alternatives) == (
        ("builtins.int",), ("builtins.float",),
    )
    assert (initial.physical_alternatives[0].dtype,
            updated.physical_alternatives[0].dtype) == ("int64", "float64")
    assert CorrelationTable._loop_native_scalar_union_gap_findings(
        SimpleNamespace(metadata={"identity_book": book})) == []
