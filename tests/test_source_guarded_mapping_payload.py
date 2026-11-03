"""The keyed value belongs to the present arm of an exact `is None` guard."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CALLABLE_IDENTITY, CANONICAL_VALUE, NATIVE_EXCEPTION_CONTROL_GAP, REDUCTION,
    SOURCE_ACTIVITY_EDGE, SOURCE_GUARDED_MAPPING_PAYLOAD,
    SOURCE_MAPPING_GET_READ, SOURCE_OPERATOR_KIND,
    SOURCE_RESOURCE_ADMISSION_DEMAND,
)
from src.compiler.exception_graph import (
    MissingResourceControlTopology, canonical_resource_call,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


def _repro(source):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        scope = graph.function_table.entry('consume').graph.G.graph[
            'authored_function_identity'
        ]
        return book, scope
    finally:
        end_identity_book(token)


def test_present_payload_cites_exact_get_guard_assignment_and_arm():
    book, scope = _repro(SOURCE)
    rows = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    assert len(rows) == 1
    row = rows[0]
    assert row[1:] == (5, 11)
    fact = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).latest(row)
    assert fact[0].page == SOURCE_MAPPING_GET_READ
    assert book.page(SOURCE_OPERATOR_KIND).cell(
        next(cell for cell in fact if cell.page == SOURCE_OPERATOR_KIND).row, 0
    ) == 'compare:Is'
    assert not CorrelationTable._source_guarded_mapping_payload_findings(
        SimpleNamespace(metadata={'identity_book': book})
    )


def test_resource_finalizer_queries_canonical_call_identity():
    book, scope = _repro(SOURCE)
    gap_row, = tuple(book.page(NATIVE_EXCEPTION_CONTROL_GAP).rows())
    gap = book.latest_ref(NATIVE_EXCEPTION_CONTROL_GAP, gap_row)
    kind, requirements = book.page(NATIVE_EXCEPTION_CONTROL_GAP).latest(gap_row)
    assert kind == 'With'
    assert requirements == ('resource_enter', 'resource_exit', 'cleanup')
    canonical_call = book.latest_ref(CANONICAL_VALUE,
        next(row for row in book.page(CANONICAL_VALUE).rows()
             if row[1] == 11 and 'consume' in str(row[0])))
    assert canonical_call in {source for source, _stage in book.edges_into(gap)}


def test_resource_call_bridge_refuses_missing_or_ambiguous_canonical_rows():
    book, scope = _repro(SOURCE)
    demand_row, = tuple(book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).scope_rows(scope))
    call = book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).latest(demand_row)[2][0][0]
    canonical = next(book.latest_ref(CANONICAL_VALUE, row)
                     for row in book.page(CANONICAL_VALUE).rows()
                     if row[1] == 11 and 'consume' in str(row[0]))
    graph = SimpleNamespace(G=SimpleNamespace(graph={
        'lexical_read_scope': canonical.row[0],
    }))
    assert canonical_resource_call(book, graph, call) == (canonical, None)
    missing = SimpleNamespace(G=SimpleNamespace(graph={
        'lexical_read_scope': ('unrelated', 0),
    }))
    assert canonical_resource_call(book, missing, call) == (
        None, 'missing_canonical_call_identity',
    )
    book.post(CANONICAL_VALUE, (canonical.row[0], 999), ('alternate',),
              stage=REDUCTION, provenance=Derived((call,)), mode=Mode.CONCORD)
    assert canonical_resource_call(book, graph, call) == (
        None, 'ambiguous_canonical_call_identity',
    )


def test_whole_table_guarded_evidence_flush_is_idempotent():
    from src.compiler.source_guarded_payload import post_guarded_mapping_payloads
    from src.compiler.source_guarded_callable import post_guarded_callables
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(SOURCE))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        before = tuple(book.clock)
        guarded_rows = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).rows())
        callable_rows = tuple(book.page(CALLABLE_IDENTITY).rows())
        post_guarded_mapping_payloads(book)
        post_guarded_callables(graph, book)
        assert tuple(book.clock) == before
        assert tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).rows()) == guarded_rows
        assert tuple(book.page(CALLABLE_IDENTITY).rows()) == callable_rows
    finally:
        end_identity_book(token)


@pytest.mark.parametrize('changed', [
    SOURCE.replace('if book is None:', 'if book == None:'),
    SOURCE.replace('        return 0\n    with book.borrowed',
                   '        marker = 0\n    with book.borrowed'),
])
def test_equal_or_nonterminal_guard_has_no_present_payload(changed):
    book, scope = _repro(changed)
    assert not tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))


def test_audit_rejects_a_later_false_arm_successor():
    book, scope = _repro(SOURCE)
    row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    fact = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).latest(row)
    terminal = next(cell for cell in fact if cell.page.name == 'source_terminal_return')
    return_exit = book.page(terminal.page).cell(terminal.row, terminal.column)[1]
    continuation = next(cell for cell in fact
                        if cell.page == SOURCE_ACTIVITY_EDGE
                        and cell.row[3] == 'flow'
                        and cell.row[2].row[2] == 'enter'
                        and book.page(cell.row[2].page).cell(
                            cell.row[2].row, cell.row[2].column)[0] == 'With')
    with_enter = continuation.row[2]
    book.post(SOURCE_ACTIVITY_EDGE, (scope, return_exit, with_enter, 'flow'),
              True, stage=REDUCTION,
              provenance=Derived((return_exit, with_enter)), mode=Mode.CONCORD)
    assert CorrelationTable._source_guarded_mapping_payload_findings(
        SimpleNamespace(metadata={'identity_book': book})
    )
