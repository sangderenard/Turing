"""A one-output call retains its callee's closed semantic class arms."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import (
    SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES, SOURCE_CALL_RESULT_IDENTITY,
    SOURCE_CLASS_UNION_NATIVE_GAP, VALUE_TYPE_ABI,
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
ROOT = '''
def root(flag: bool) -> int:
    return choose(flag).value()
'''


@pytest.mark.parametrize("reversed_functions", (False, True))
def test_call_result_retains_closed_classes_without_native_abi(
    monkeypatch, reversed_functions,
):
    source = CLASSES + (ROOT + CHOOSE if reversed_functions
                        else CHOOSE + ROOT)
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
            source, "root", name="call_closed_classes",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None,
        )
    assert len(captured) == 1
    book = captured[0]
    rows = tuple(book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).rows())
    assert len(rows) == 1
    row = rows[0]
    assert row[1:] == (1, row[2], 5)
    transport = book.latest_ref(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES, row)
    assert transport.column == 0  # fixed-point rounds did not revise the fact
    output, returned, closed, reached, call = book.page(
        SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cell(
            transport.row, transport.column)
    assert (output.row[1], returned.row[1], closed.row[1],
            reached.row[1], call.row[1]) == (0, 5, 5, 1, 1)
    answer = require_class_alternatives(row[0], 1,
                                        control_scope=call, book=book)
    assert isinstance(answer, ClassAlternativesEvidence)
    assert answer.alternatives == (("A", 1), ("B", 1))
    assert answer.control_scope == call
    assert closed in answer.proof_cells
    assert isinstance(require_class_alternatives(row[0], 1, book=book),
                      UnresolvedSourceDemand)
    assert book.latest_ref(SOURCE_CALL_RESULT_IDENTITY, row[:2]) is None
    assert book.latest_ref(VALUE_TYPE_ABI, row[:2]) is None
    assert CorrelationTable._source_call_closed_class_alternatives_findings(
        SimpleNamespace(metadata={"identity_book": book})) == []
