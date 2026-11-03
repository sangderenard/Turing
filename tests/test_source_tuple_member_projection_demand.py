"""An exact tuple actual does not make its member a callee-local value."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    REDUCTION, SOURCE_AGGREGATE_LAYOUT, SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, SOURCE_VALUE_CLASS,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.compiler.source_evidence_resolver import (
    UnresolvedSourceDemand, require_member,
)
from src.compiler.source_tuple_member_demand import post_tuple_member_demands
from src.compiler import source_evidence_flush
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = '''\
def consume(pair):
    return pair[0]

def root():
    return consume((1, 2))
'''


def _reduced(source=SOURCE):
    book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return book, token, graph


def test_exact_call_tuple_member_is_named_unresolved_and_idempotent():
    book, token, graph = _reduced()
    try:
        root = graph.function_table.entry('root').graph.G
        consume = graph.function_table.entry('consume').graph.G
        root_scope = root.graph['authored_function_identity']
        consume_scope = consume.graph['authored_function_identity']
        actual = book.latest_ref(SOURCE_AGGREGATE_LAYOUT,
                                 (root.graph['source_aggregate_layout_scope'], 2))
        assert actual is not None
        assert len(book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            actual.row, actual.column).members) == 2
        assert book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL,
                               (root_scope, 3, consume_scope, 0)) is not None
        assert book.page(SOURCE_VALUE_CLASS).latest((consume_scope, 0))[0] == 'builtins.tuple'
        assert book.latest_ref(SOURCE_AGGREGATE_LAYOUT,
                               (consume.graph['source_aggregate_layout_scope'], 0)) is None
        unresolved = require_member(
            consume.graph['source_aggregate_layout_scope'], 0, 0, book=book)
        assert isinstance(unresolved, UnresolvedSourceDemand)
        assert unresolved.missing == 'missing_aggregate_layout'
        page = book.page(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND)
        row, = tuple(page.rows())
        assert row == (root_scope, 3, consume_scope, 0, 2, 0)
        fact = page.latest(row)
        assert fact[0] == 'callee_member_identity_unpublished'
        assert fact[6].page == SOURCE_AGGREGATE_LAYOUT
        assert not CorrelationTable._source_call_tuple_member_projection_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
        before = tuple(book.clock)
        post_tuple_member_demands(tuple(reversed(tuple(graph.function_table))),
                                  book=book)
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_tuple_member_demand_audit_rejects_wrong_member_cell():
    book, token, graph = _reduced()
    try:
        page = book.page(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND)
        row, = tuple(page.rows())
        cell = book.latest_ref(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, row)
        fact = page.cell(cell.row, cell.column)
        wrong = (*fact[:7], fact[3], *fact[8:])
        book.post(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, row, wrong,
                  stage=REDUCTION, provenance=Derived((cell, fact[3])),
                  mode=Mode.REVISE)
        assert CorrelationTable._source_call_tuple_member_projection_demand_findings(
            SimpleNamespace(metadata={'identity_book': book},
                            function_table=graph.function_table))
    finally:
        end_identity_book(token)


def test_deferred_flush_reversed_order_has_same_demand_and_repeat_is_stable(monkeypatch):
    original = source_evidence_flush.flush_source_evidence
    snapshots = []
    for reverse in (False, True):
        monkeypatch.setattr(source_evidence_flush, 'flush_source_evidence',
                            lambda _graph: 0)
        book, token, graph = _reduced()
        monkeypatch.setattr(source_evidence_flush, 'flush_source_evidence',
                            original)
        try:
            if reverse:
                table = graph.function_table
                graph.function_table = type('ReversedTable', (), {
                    '__iter__': lambda _self: iter(tuple(reversed(tuple(table))))
                })()
            assert original(graph, book=book) >= 1
            page = book.page(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND)
            row, = tuple(page.rows())
            cell = book.latest_ref(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            snapshots.append((row[1], row[3:], fact[0],
                              tuple(item.page.name for item in fact[1:]),
                              tuple(sorted(source.page.name for source, _stage
                                           in book.edges_into(cell)))))
            before = tuple(book.clock)
            assert original(graph, book=book) == 0
            assert tuple(book.clock) == before
        finally:
            end_identity_book(token)
    assert snapshots[0] == snapshots[1]
