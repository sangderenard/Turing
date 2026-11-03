"""A returned formal forwards only a complete semantic class domain."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import (
    SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_CLASS_UNION_NATIVE_GAP,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
    VALUE_TYPE_ABI,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, IdentityBook
from src.compiler.source_class_domain_forward import (
    post_formal_return_class_domains,
)
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
RELAY = '''
def relay(value):
    return value
'''


def _captured_book(monkeypatch, source):
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
            source, "root", name="closed_class_relay",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None,
        )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("reversed_functions", (False, True))
def test_exact_returned_formal_forwards_closed_classes(
    monkeypatch, reversed_functions,
):
    root = '''
def root(flag: bool):
    item = choose(flag)
    return relay(item)
'''
    source = CLASSES + (root + RELAY + CHOOSE if reversed_functions
                        else CHOOSE + RELAY + root)
    book = _captured_book(monkeypatch, source)
    rows = tuple(book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).rows())
    assert len(rows) == 2
    relay_row = next(row for row in rows if row[2][1] == "relay")
    cell = book.latest_ref(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
                           relay_row)
    assert cell.column == 0
    output, returned, domain, reached, call = book.page(
        SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cell(cell.row, cell.column)
    assert domain.page == SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES
    assert (output.row[1], returned.row[1], reached.row[1],
            call.row[1]) == (0, 0, relay_row[1], relay_row[1])
    answer = require_class_alternatives(
        relay_row[0], relay_row[1], control_scope=call, book=book,
    )
    assert isinstance(answer, ClassAlternativesEvidence)
    assert answer.alternatives == (("A", 1), ("B", 1))
    assert domain in answer.proof_cells
    assert isinstance(require_class_alternatives(
        relay_row[0], relay_row[1], book=book,
    ), UnresolvedSourceDemand)
    assert book.latest_ref(VALUE_TYPE_ABI, relay_row[:2]) is None
    assert post_formal_return_class_domains(book) == 0
    assert book.latest_ref(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
                           relay_row) == cell
    assert CorrelationTable._source_call_closed_class_alternatives_findings(
        SimpleNamespace(metadata={"identity_book": book})) == []


def test_second_relay_call_keeps_return_domain_open(monkeypatch):
    root = '''
def root(flag: bool):
    item = choose(flag)
    relay(A())
    return relay(item)
'''
    book = _captured_book(monkeypatch,
                          CLASSES + CHOOSE + RELAY + root)
    rows = tuple(book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).rows())
    assert len(rows) == 1
    assert rows[0][2][1] == "choose"
    formal_row, = book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).rows()
    answer = require_class_alternatives(formal_row[0], formal_row[1],
                                        book=book)
    assert isinstance(answer, UnresolvedSourceDemand)
    assert answer.missing == "incomplete_formal_closed_class_domain"
