"""A present mapping payload may name its authored method address exactly."""

import ast
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CALLABLE_IDENTITY, CLASS_METHOD_DECLARATION, FUNCTION_ADDRESS,
    NATIVE_EXCEPTION_CONTROL_GAP, SOURCE_GUARDED_MAPPING_PAYLOAD,
    SOURCE_SPAN, SOURCE_VALUE_CLASS,
)
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.source_guarded_callable import guarded_callable_facts, post_guarded_callables
from src.transmogrifier.function_table import function_address_identity
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from tests.test_cross_function_mapping_resource_evidence import SOURCE


def _repro(source=SOURCE):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        return book, graph
    finally:
        end_identity_book(token)


def test_guarded_borrowed_call_uses_exact_authored_method_span_and_address():
    book, graph = _repro()
    scope = graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity']
    guard_row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    callable_cell = book.latest_ref(CALLABLE_IDENTITY, (scope, guard_row[2]))
    assert callable_cell is not None
    assert book.page(CALLABLE_IDENTITY).cell(callable_cell.row, callable_cell.column) == (
        graph.function_table.entry('IdentityBook.borrowed').reference.address)
    direct = {source for source, _stage in book.edges_into(callable_cell)}
    method, = (source for source in direct
               if source.page == CLASS_METHOD_DECLARATION)
    address, = (source for source in direct if source.page == FUNCTION_ADDRESS)
    method_spans = {source for source, _stage in book.edges_into(method)
                    if source.page == SOURCE_SPAN}
    address_spans = {source for source, _stage in book.edges_into(address)
                     if source.page == SOURCE_SPAN}
    assert len(method_spans & address_spans) == 1
    assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, guard_row[1])) is None
    assert not CorrelationTable._source_guarded_callable_findings(
        SimpleNamespace(metadata={'identity_book': book},
                        function_table=graph.function_table))

    # The publisher is a fixed point and remains valid if the function table
    # is enumerated in reverse order.
    address_history = tuple(book.page(FUNCTION_ADDRESS).history(address.row))
    callable_history = tuple(book.page(CALLABLE_IDENTITY).history(callable_cell.row))
    reversed_graph = SimpleNamespace(
        function_table=tuple(reversed(tuple(graph.function_table))), G=graph.G)
    post_guarded_callables(reversed_graph, book)
    post_guarded_callables(reversed_graph, book)
    assert tuple(book.page(FUNCTION_ADDRESS).history(address.row)) == address_history
    assert tuple(book.page(CALLABLE_IDENTITY).history(callable_cell.row)) == callable_history


def test_nonterminal_guard_has_no_guarded_callable():
    source = SOURCE.replace('        return 0\n    with book.borrowed',
                            '        marker = 0\n    with book.borrowed')
    book, graph = _repro(source)
    scope = graph.function_table.entry('consume').graph.G.graph[
        'authored_function_identity']
    assert not tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    assert not tuple(book.page(CALLABLE_IDENTITY).scope_rows(scope))


def test_guarded_method_uses_authored_identity_when_navigation_label_is_qualified():
    book, graph = _repro()
    target = graph.function_table.entry('IdentityBook.borrowed')
    authored = tuple(target.graph.G.graph['authored_function_identity'])
    qualified = replace(
        target, qualified_name='.'.join(authored),
        metadata={**target.metadata,
                  'function_address_identity': function_address_identity(target)},
    )
    assert qualified.qualified_name != target.qualified_name
    view = SimpleNamespace(
        function_table=tuple(qualified if entry is target else entry
                             for entry in graph.function_table), G=graph.G,
        metadata={'identity_book': book},
    )
    scope = graph.function_table.entry('consume').graph.G.graph['authored_function_identity']
    row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    guard = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, row)
    facts = guarded_callable_facts(view, book, row, guard)
    assert facts is not None and facts[0] == target.reference.address
    assert not CorrelationTable._source_guarded_callable_findings(view)

    # The same navigation label and address cannot redeem another source
    # module's authored identity.
    target.graph.G.graph['authored_function_identity'] = (
        authored[0] + '.foreign', authored[1],
    )
    try:
        assert guarded_callable_facts(view, book, row, guard) is None
        assert CorrelationTable._source_guarded_callable_findings(view)
    finally:
        target.graph.G.graph['authored_function_identity'] = authored


def test_audit_rejects_a_different_accessor_on_the_guarded_call():
    book, graph = _repro()
    consume = graph.function_table.entry('consume').graph
    scope = consume.G.graph['authored_function_identity']
    row, = tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).scope_rows(scope))
    expression = consume.G.nodes[row[2]]['expr_obj']
    assert isinstance(expression, ast.Call)
    expression.func.attr = 'other_method'
    assert CorrelationTable._source_guarded_callable_findings(
        SimpleNamespace(metadata={'identity_book': book},
                        function_table=graph.function_table))
