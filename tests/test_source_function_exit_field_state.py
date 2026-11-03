"""A formal's final keyed field write has an exact successful-exit receipt."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    REDUCER_FIELD_STATE,
    SOURCE_ACTIVITY_NODE,
    SOURCE_FORMAL_ORDER,
    SOURCE_CALL_FIELD_EFFECT_COMPLETION,
    SOURCE_FUNCTION_EXIT_FIELD_STATE,
    FieldStateKind,
)
from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _reduce(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return graph


def test_straight_line_keyed_write_publishes_formal_field_state_at_exit():
    book, token = begin_identity_book()
    try:
        graph = _reduce(
            "class Module:\n"
            "    def __init__(self):\n"
            "        self.metadata = {}\n"
            "def publish(module: Module):\n"
            "    module.metadata['key'] = 7\n"
            "def root():\n"
            "    module = Module()\n"
            "    publish(module)\n"
            "    return 1\n"
        )
        function = graph.function_table.entry("publish").graph.G
        scope = function.graph["authored_function_identity"]
        page = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE)
        rows = tuple(row for row in page.rows() if row[0] == scope)
        assert len(rows) == 2
        row = next(row for row in rows if page.latest(row)[0].page == CANONICAL_VALUE)
        assert row[2] == "metadata"
        cell = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE, row)
        formal, state, exit_activity, assignment_source, flow_edge = page.cell(
            cell.row, cell.column,
        )
        assert formal.page == CANONICAL_VALUE
        assert formal.row[-1] == row[1]
        assert state.page == REDUCER_FIELD_STATE
        assert state.row[1:] == (row[1], "metadata")
        assert book.page(REDUCER_FIELD_STATE).cell(
            state.row, state.column,
        ).kind is FieldStateKind.ELEMENT_WRITTEN
        assert exit_activity.page == SOURCE_ACTIVITY_NODE
        assert exit_activity.row[-1] == "exit"
        sources = {source for source, _stage in book.edges_into(cell)}
        assert {formal, state, exit_activity,
                assignment_source, flow_edge} <= sources
        prior = next(source for source in sources
                     if source.page == SOURCE_FUNCTION_EXIT_FIELD_STATE)
        assert book.latest_ref(SOURCE_FORMAL_ORDER, (scope,)) in {
            source for source, _stage in book.edges_into(prior)
        }
        assert CorrelationTable._source_function_exit_field_state_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        ) == []
    finally:
        end_identity_book(token)


def test_branching_field_write_has_no_straight_line_exit_receipt():
    book, token = begin_identity_book()
    try:
        graph = _reduce(
            "class Module:\n"
            "    def __init__(self):\n"
            "        self.metadata = {}\n"
            "def publish(module: Module, flag):\n"
            "    if flag:\n"
            "        module.metadata['key'] = 7\n"
            "def root(flag):\n"
            "    module = Module()\n"
            "    publish(module, flag)\n"
            "    return 1\n"
        )
        scope = graph.function_table.entry("publish").graph.G.graph[
            "authored_function_identity"
        ]
        assert not tuple(row for row in book.page(
            SOURCE_FUNCTION_EXIT_FIELD_STATE,
        ).rows() if row[0] == scope)
        assert not tuple(row for row in book.page(
            SOURCE_CALL_FIELD_EFFECT_COMPLETION,
        ).rows() if row[2] == scope)
    finally:
        end_identity_book(token)
