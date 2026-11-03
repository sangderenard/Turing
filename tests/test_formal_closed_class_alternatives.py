"""Closed source class arms cross only complete actual/formal inventories."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import (
    SOURCE_CALL_ACTUAL_FORMAL, SOURCE_CLASS_UNION_NATIVE_GAP,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES, SOURCE_VALUE_CLASS,
    VALUE_TYPE_ABI,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, IdentityBook
from src.compiler.source_evidence_resolver import (
    ClassAlternativesEvidence, UnresolvedSourceDemand,
    require_class_alternatives,
)


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)
CLASSES = '''
class A:
    def value(self) -> int: return 1
class B:
    def value(self) -> int: return 2
'''
CHOOSE = '''
def choose(flag: bool):
    return A() if flag else B()
'''
CONSUME = '''
def consume(value):
    return value.value()
'''


def _lower_and_capture(monkeypatch, source):
    captured = []
    original = IdentityBook.post

    def capture(self, page, row, fact, *args, **kwargs):
        result = original(self, page, row, fact, *args, **kwargs)
        if page == SOURCE_CLASS_UNION_NATIVE_GAP:
            captured.append(self)
        return result

    monkeypatch.setattr(IdentityBook, "post", capture)
    with pytest.raises(Exception, match="native_class_union_arm_layout_missing"):
        lower_ast_source_to_ssa(
            source, "root", name="formal_closed_classes",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None,
        )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("reversed_functions", (False, True))
def test_formal_receives_closed_classes_across_exact_call_edges(
    monkeypatch, reversed_functions,
):
    root = '''
def root(flag: bool) -> int:
    item = choose(flag)
    return consume(item)
'''
    source = CLASSES + (root + CONSUME + CHOOSE if reversed_functions
                        else CHOOSE + CONSUME + root)
    book = _lower_and_capture(monkeypatch, source)
    rows = tuple(book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).rows())
    assert len(rows) == 1
    row = rows[0]
    transport = book.latest_ref(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
                                row)
    assert transport.column == 0
    binding, call_domain, closed, output, formal_owner = book.page(
        SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).cell(
            transport.row, transport.column)
    assert binding.page == SOURCE_CALL_ACTUAL_FORMAL
    assert call_domain.row[0] == row[2]
    assert (closed.row[1], output.row[1], formal_owner.row[1]) == (
        5, 0, "value",
    )
    answer = require_class_alternatives(row[0], row[1], book=book)
    assert isinstance(answer, ClassAlternativesEvidence)
    assert answer.alternatives == (("A", 1), ("B", 1))
    assert {binding, call_domain, closed, output, formal_owner} <= set(
        answer.proof_cells)
    assert answer.control_scope == formal_owner
    assert book.latest_ref(SOURCE_VALUE_CLASS, row[:2]) is None
    assert book.latest_ref(VALUE_TYPE_ABI, row[:2]) is None
    assert CorrelationTable._source_formal_closed_class_alternatives_findings(
        SimpleNamespace(metadata={"identity_book": book})) == []


def test_second_unproved_call_keeps_formal_domain_open(monkeypatch):
    root = '''
def root(flag: bool) -> int:
    item = choose(flag)
    consume(A())
    return consume(item)
'''
    book = _lower_and_capture(monkeypatch,
                              CLASSES + CHOOSE + CONSUME + root)
    (row,) = book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).rows()
    # The second reached actual/formal row has no closed call-result page.
    assert len(tuple(candidate for candidate in
                     book.page(SOURCE_CALL_ACTUAL_FORMAL).rows()
                     if candidate[3] == row[1]
                     and candidate[2][1] == "consume")) == 2
    answer = require_class_alternatives(row[0], row[1], book=book)
    assert isinstance(answer, UnresolvedSourceDemand)
    assert answer.missing == "incomplete_formal_closed_class_domain"
