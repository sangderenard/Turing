"""A mapping payload's class is valid only in its guarded call arm."""

from src.compiler.concordance_declarations import (
    SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_VALUE_CLASS,
)
from src.compiler.source_evidence_resolver import (
    ClassEvidence, UnresolvedSourceDemand, require_class,
)
from tests.test_cross_function_mapping_resource_evidence import SOURCE
from tests.test_source_guarded_callable import _repro


def test_present_arm_class_uses_stored_constructor_without_global_get_class():
    book, graph = _repro()
    scope = graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity']
    row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    guard = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, row)
    before = tuple(book.clock)
    evidence = require_class(scope, row[1], control_scope=guard, book=book)
    assert isinstance(evidence, ClassEvidence)
    assert evidence.class_identity == 'IdentityBook'
    assert evidence.value_id == row[1]
    assert evidence.control_scope == guard
    assert evidence.class_cell.page == SOURCE_VALUE_CLASS
    assert evidence.class_cell.row[0][1] == 'publish'
    assert guard in evidence.proof_cells
    assert evidence.class_cell in evidence.proof_cells
    assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, row[1])) is None
    global_result = require_class(scope, row[1], book=book)
    assert isinstance(global_result, UnresolvedSourceDemand)
    assert global_result.missing == 'missing_source_value_class'
    assert tuple(book.clock) == before


def test_guarded_class_rejects_a_different_value_or_nonterminal_guard():
    book, graph = _repro()
    scope = graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity']
    row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    guard = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, row)
    wrong_value = require_class(scope, row[2], control_scope=guard, book=book)
    assert isinstance(wrong_value, UnresolvedSourceDemand)
    assert wrong_value.missing == 'missing_exact_guarded_scope'

    source = SOURCE.replace('        return 0\n    with book.borrowed',
                            '        marker = 0\n    with book.borrowed')
    absent_book, absent_graph = _repro(source)
    absent_scope = absent_graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity']
    assert not tuple(absent_book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(
        absent_scope))
    absent = require_class(absent_scope, row[1], book=absent_book)
    assert isinstance(absent, UnresolvedSourceDemand)
    assert absent.missing == 'missing_source_value_class'
