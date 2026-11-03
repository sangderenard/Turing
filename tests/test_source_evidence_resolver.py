import ast

import pytest

from src.common.tensors.topological_reducer import (
    _source_numeric_scope,
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    CALLABLE_IDENTITY,
    FUNCTION_ADDRESS,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.source_evidence_resolver import (
    CallableEvidence,
    UnresolvedSourceDemand,
    require_callable,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_dynamic_resource_call_reads_exact_published_callable_and_inputs():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "class IdentityBook:\n"
            "    def borrowed(self, name):\n"
            "        return name\n"
            "def identity_book(module) -> IdentityBook:\n"
            "    return IdentityBook()\n"
            "def root(module, names):\n"
            "    book = identity_book(module)\n"
            "    for name in names:\n"
            "        with book.borrowed(name) as page:\n"
            "            return page\n"
        ))
        with pytest.raises(MissingResourceControlTopology) as refusal:
            reduce_abstract_tensor_topology(graph)
        assert "entry, exit and cleanup" in str(refusal.value)
        function = graph.function_table.entry("root").graph
        call = next(
            node for statement in function.G.graph["function_body"]
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "borrowed"
        )
        scope = _source_numeric_scope(function)
        before = tuple(book.clock)
        answer = require_callable(function, scope, id(call), book=book)
        assert isinstance(answer, CallableEvidence)
        assert answer.callable_cell == book.latest_ref(
            CALLABLE_IDENTITY, (scope, id(call)),
        )
        assert answer.address_cell.page == FUNCTION_ADDRESS
        assert answer.address == book.page(FUNCTION_ADDRESS).cell(
            answer.address_cell.row, answer.address_cell.column,
        )
        assert answer.proof_cells
        assert tuple(role for role, _cell in answer.ordered_inputs) == (
            "operand", "callee", "arg:0",
        )
        assert any(role == "operand" for role, _cell in answer.ordered_inputs)
        assert answer.control_scope.page.name == "source_activity_binding"
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_unproved_call_returns_named_demand_without_publishing_a_guess():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "def root(manager):\n"
            "    return manager()\n"
        )
        graph.build_from_ast(tree)
        call = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Call))
        scope = _source_numeric_scope(graph)
        before = tuple(book.clock)
        answer = require_callable(graph, scope, id(call), book=book)
        assert isinstance(answer, UnresolvedSourceDemand)
        assert answer.facet == "callable"
        assert answer.missing == "missing_callable_identity"
        assert book.latest_ref(CALLABLE_IDENTITY, (scope, id(call))) is None
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)
