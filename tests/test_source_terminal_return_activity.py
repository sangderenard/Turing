"""A source Return cuts only its unprotected normal activity successor."""

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE, SOURCE_TERMINAL_RETURN,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _activity(source):
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
        scope = graph.function_table.entry('root').graph.G.graph[
            'authored_function_identity'
        ]
        assert not CorrelationTable._control_activity_findings(
            SimpleNamespace(metadata={'identity_book': book})
        )
        return book, scope
    finally:
        end_identity_book(token)


def _nodes(book, scope, kind, phase):
    return tuple(book.latest_ref(SOURCE_ACTIVITY_NODE, row)
                 for row in book.page(SOURCE_ACTIVITY_NODE).scope_rows(scope)
                 if row[2] == phase
                 and book.page(SOURCE_ACTIVITY_NODE).latest(row)[0] == kind)


def _reachable(book, scope, start, end):
    edges = tuple((row[1], row[2]) for row in
                  book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope))
    pending, seen = [start], set()
    while pending:
        node = pending.pop()
        if node == end:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(right for left, right in edges if left == node)
    return False


def test_none_guard_return_cannot_reach_following_with():
    source = '''
def root(book):
    if book is None:
        return 0
    with book.borrowed('name') as page:
        return 1
'''
    # The generic context has no resource contract. Its refusal occurs after
    # source activity was posted, so inspect the existing book at that cut.
    from src.compiler.exception_graph import MissingResourceControlTopology
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(source))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        scope = graph.function_table.entry('root').graph.G.graph[
            'authored_function_identity'
        ]
        terminal = tuple(book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope))
        assert len(terminal) == 1
        return_exit = book.page(SOURCE_TERMINAL_RETURN).latest(terminal[0])[1]
        with_enter, = _nodes(book, scope, 'With', 'enter')
        assert not _reachable(book, scope, return_exit, with_enter)
        assert not [row for row in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                    if row[1] == return_exit]
        assert not CorrelationTable._control_activity_findings(
            SimpleNamespace(metadata={'identity_book': book})
        )
    finally:
        end_identity_book(token)


def test_direct_return_does_not_reach_unreachable_statement():
    book, scope = _activity('''
def root():
    return 1
    marker = 2
''')
    terminal, = tuple(book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope))
    return_exit = book.page(SOURCE_TERMINAL_RETURN).latest(terminal)[1]
    assign_enter, = _nodes(book, scope, 'Assign', 'enter')
    assert not _reachable(book, scope, return_exit, assign_enter)


def test_loop_return_does_not_recur_to_header():
    book, scope = _activity('''
def root(flag):
    while flag:
        return 1
    return 0
''')
    terminals = tuple(book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope))
    assert len(terminals) == 2
    for row in terminals:
        return_exit = book.page(SOURCE_TERMINAL_RETURN).latest(row)[1]
        assert not [edge for edge in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                    if edge[1] == return_exit]


def test_audit_rejects_spurious_return_successor():
    from src.compiler.concordance_declarations import REDUCTION
    book, scope = _activity('''
def root():
    return 1
    marker = 2
''')
    terminal, = tuple(book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope))
    return_exit = book.page(SOURCE_TERMINAL_RETURN).latest(terminal)[1]
    assign_enter, = _nodes(book, scope, 'Assign', 'enter')
    book.post(SOURCE_ACTIVITY_EDGE,
              (scope, return_exit, assign_enter, 'flow'), True,
              stage=REDUCTION, provenance=Derived((return_exit, assign_enter)),
              mode=Mode.CONCORD)
    assert any(item.kind == 'source-terminal-return' for item in
               CorrelationTable._control_activity_findings(
                   SimpleNamespace(metadata={'identity_book': book})))


def test_converted_resource_return_keeps_cleanup_activity():
    from src.compiler.concordance_declarations import SOURCE_RESOURCE_CONVERSION
    from src.transmogrifier.graph.python_special_cases import lower_python_threading
    source = '''
import threading
def root():
    lock = threading.Lock()
    with lock:
        return 1
'''
    book, token = begin_identity_book()
    try:
        tree = ast.parse(source)
        constructor = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'Lock')
        constructor._turing_dispatch_operation = {
            'operation': 'sync_mutex_create',
            'source_identity': 'threading.Lock',
        }
        converted = lower_python_threading(tree)
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(converted)
        reduce_abstract_tensor_topology(graph)
        scope = graph.function_table.entry('root').graph.G.graph[
            'authored_function_identity'
        ]
        assert tuple(book.page(SOURCE_RESOURCE_CONVERSION).rows())
        assert not tuple(book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope))
        edges = tuple(book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope))
        assert any(row[3] == 'resource_cleanup' for row in edges)
        assert not CorrelationTable._control_activity_findings(
            SimpleNamespace(metadata={'identity_book': book})
        )
    finally:
        end_identity_book(token)
