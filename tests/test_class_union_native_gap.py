"""Exact authored class arms without physical constructor cells refuse early."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import SOURCE_CLASS_UNION_NATIVE_GAP
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, IdentityBook


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)
SOURCE = '''
class A:
    def value(self) -> int: return 1
class B:
    def value(self) -> int: return 2
def choose(flag: bool):
    return A() if flag else B()
def root(flag: bool) -> int:
    return choose(flag).value()
'''


def test_closed_class_constructor_merge_names_missing_native_arm_layout(monkeypatch):
    captured = []
    original = IdentityBook.post

    def capture(self, page, row, fact, *args, **kwargs):
        result = original(self, page, row, fact, *args, **kwargs)
        if page == SOURCE_CLASS_UNION_NATIVE_GAP:
            captured.append((self, result))
        return result

    monkeypatch.setattr(IdentityBook, "post", capture)
    with pytest.raises(
        Exception,
        match=r"native_class_union_arm_layout_missing: merge=5, arms=\(3, 4\), classes=\('A', 'B'\)",
    ):
        lower_ast_source_to_ssa(
            SOURCE, "root", name="closed_class_union_native_gap",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None,
        )
    assert len(captured) == 1
    book, gap = captured[0]
    fact = book.page(SOURCE_CLASS_UNION_NATIVE_GAP).cell(
        gap.row, gap.column)
    assert fact.alternatives == ("A", "B")
    assert tuple(arm.row[1] for arm in fact.arm_value_cells) == (3, 4)
    assert CorrelationTable._source_class_union_native_gap_findings(
        SimpleNamespace(metadata={"identity_book": book})) == []
