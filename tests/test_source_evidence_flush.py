"""The whole-source evidence family settles deterministically before admission."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler import source_evidence_flush
from src.compiler.concordance_declarations import (
    CALLABLE_IDENTITY, REDUCTION, SOURCE_EVIDENCE_FLUSH_ROUND,
    SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_MAPPING_GET_READ, SOURCE_SPAN,
)
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


class ReversedTable:
    def __init__(self, table):
        self.table = table

    def __iter__(self):
        return iter(tuple(reversed(tuple(self.table))))


def _deferred_repro(monkeypatch):
    book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(SOURCE))
    original = source_evidence_flush.flush_source_evidence
    monkeypatch.setattr(source_evidence_flush, 'flush_source_evidence',
                        lambda _graph: 0)
    try:
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
    finally:
        monkeypatch.setattr(source_evidence_flush, 'flush_source_evidence',
                            original)
    assert tuple(book.page(SOURCE_MAPPING_GET_READ).rows()) == ()
    return graph, book, token


def _settled(book):
    read_rows = tuple(book.page(SOURCE_MAPPING_GET_READ).rows())
    guard_rows = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).rows())
    callable_rows = tuple(row for row in book.page(CALLABLE_IDENTITY).rows()
                          if row[0][-1] == 'consume')
    assert len(read_rows) == len(guard_rows) == len(callable_rows) == 1
    read = book.latest_ref(SOURCE_MAPPING_GET_READ, read_rows[0])
    guard = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, guard_rows[0])
    callable_cell = book.latest_ref(CALLABLE_IDENTITY, callable_rows[0])
    assert not CorrelationTable._source_evidence_flush_round_findings(
        SimpleNamespace(metadata={'identity_book': book}))
    return (
        (read_rows[0][1:], guard_rows[0][1:], callable_rows[0][1]),
        (book.page(SOURCE_MAPPING_GET_READ).cell(read.row, read.column)[5].page.name,
         tuple(cell.page.name for cell in book.page(
             SOURCE_GUARDED_MAPPING_PAYLOAD).cell(guard.row, guard.column)),
         book.page(CALLABLE_IDENTITY).cell(callable_cell.row,
                                           callable_cell.column)),
        (tuple(sorted(source.page.name for source, _stage in book.edges_into(read))),
         tuple(sorted(source.page.name for source, _stage in book.edges_into(guard))),
         tuple(sorted(source.page.name for source, _stage in book.edges_into(
             callable_cell)))),
    )


def test_repeated_flush_adds_no_rows_or_clock_entries(monkeypatch):
    graph, book, token = _deferred_repro(monkeypatch)
    try:
        assert source_evidence_flush.flush_source_evidence(graph, book=book) == 2
        settled = _settled(book)
        before = tuple(book.clock)
        assert source_evidence_flush.flush_source_evidence(graph, book=book) == 0
        assert tuple(book.clock) == before
        assert _settled(book) == settled
        rounds = tuple(book.page(SOURCE_EVIDENCE_FLUSH_ROUND).rows())
        assert tuple(row[1] for row in rounds) == (1, 2)
        for row in rounds:
            fact = book.page(SOURCE_EVIDENCE_FLUSH_ROUND).latest(row)
            assert len(fact[0]) == len(fact[1]) == 64
            assert all(len(change) == 2 for change in fact[2])
            assert not any(isinstance(part, tuple) and len(part) > 100
                           for part in fact[:2])
    finally:
        end_identity_book(token)


def test_reversed_function_order_has_same_settled_facts_and_causes(monkeypatch):
    snapshots = []
    for reverse in (False, True):
        graph, book, token = _deferred_repro(monkeypatch)
        try:
            if reverse:
                graph.function_table = ReversedTable(graph.function_table)
            assert source_evidence_flush.flush_source_evidence(graph, book=book) == 2
            snapshots.append(_settled(book))
        finally:
            end_identity_book(token)
    assert snapshots[0] == snapshots[1]


def test_same_cell_new_provenance_edge_reopens_stable_flush(monkeypatch):
    graph, book, token = _deferred_repro(monkeypatch)
    try:
        assert source_evidence_flush.flush_source_evidence(graph, book=book) == 2
        row, = tuple(book.page(SOURCE_MAPPING_GET_READ).rows())
        read = book.latest_ref(SOURCE_MAPPING_GET_READ, row)
        fact = book.page(SOURCE_MAPPING_GET_READ).cell(read.row, read.column)
        old_sources = {source for source, _stage in book.edges_into(read)}
        extra = next(book.latest_ref(SOURCE_SPAN, span_row)
                     for span_row in book.page(SOURCE_SPAN).rows()
                     if span_row[1] == 'root'
                     and book.latest_ref(SOURCE_SPAN, span_row) not in old_sources)
        book.post(SOURCE_MAPPING_GET_READ, row, fact, stage=REDUCTION,
                  provenance=Derived((extra,)), mode=Mode.CONCORD)
        assert book.latest_ref(SOURCE_MAPPING_GET_READ, row) == read
        assert extra in {source for source, _stage in book.edges_into(read)}
        assert source_evidence_flush.flush_source_evidence(graph, book=book) == 1
        rounds = tuple(book.page(SOURCE_EVIDENCE_FLUSH_ROUND).rows())
        assert tuple(row[1] for row in rounds) == (1, 2, 3)
        assert not CorrelationTable._source_evidence_flush_round_findings(
            SimpleNamespace(metadata={'identity_book': book}))
    finally:
        end_identity_book(token)


def test_flush_round_audit_rejects_lost_result_cells(monkeypatch):
    graph, book, token = _deferred_repro(monkeypatch)
    try:
        source_evidence_flush.flush_source_evidence(graph, book=book)
        row = next(row for row in book.page(SOURCE_EVIDENCE_FLUSH_ROUND).rows()
                   if row[1] == 1)
        original = book.latest_ref(SOURCE_EVIDENCE_FLUSH_ROUND, row)
        before, after, changes, guard, restarted = book.page(
            SOURCE_EVIDENCE_FLUSH_ROUND).cell(original.row, original.column)
        book.post(SOURCE_EVIDENCE_FLUSH_ROUND, row,
                  (before, after, (), guard, restarted), stage=REDUCTION,
                  provenance=Derived((original, guard)), mode=Mode.REVISE)
        assert CorrelationTable._source_evidence_flush_round_findings(
            SimpleNamespace(metadata={'identity_book': book}))
    finally:
        end_identity_book(token)
