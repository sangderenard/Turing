"""Conflicting authored IfExp classes retain exact semantic arm evidence."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    PROCESS_GRAPH_OPERANDS, SOURCE_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_VALUE_CLASS, VALUE_TYPE_ABI,
)
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.source_evidence_resolver import (
    ClassAlternativesEvidence, UnresolvedSourceDemand, require_class,
    require_class_alternatives,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


CLASSES = (
    "class A:\n"
    "    def value(self) -> int: return 1\n"
    "class B:\n"
    "    def value(self) -> int: return 2\n"
)


def _choice(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    function = graph.function_table.entry("choose").graph.G
    choice = next(int(node_id) for node_id, data in function.nodes(data=True)
                  if isinstance(data.get("expr_obj"), ast.IfExp))
    return function, function.graph["source_aggregate_layout_scope"], choice


def test_distinct_authored_classes_form_control_owned_semantic_alternatives():
    book, token = begin_identity_book()
    try:
        _function, scope, choice = _choice(
            CLASSES + "def choose(flag: bool):\n"
            "    return A() if flag else B()\n")
        cell = book.latest_ref(SOURCE_CLOSED_CLASS_ALTERNATIVES,
                               (scope, choice))
        assert cell is not None
        fact = book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cell(
            cell.row, cell.column)
        assert fact.alternatives == (("A", 1), ("B", 1))
        operands = book.page(PROCESS_GRAPH_OPERANDS).cell(
            fact.operand_cell.row, fact.operand_cell.column)
        assert tuple((ref, role) for ref, role in operands
                     if role in ("body", "orelse")) == (
                         (fact.arm_value_cells[0], "body"),
                         (fact.arm_value_cells[1], "orelse"))
        assert all(ref.page == SOURCE_VALUE_CLASS
                   for ref in fact.arm_class_cells)
        assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, choice)) is None
        assert book.latest_ref(VALUE_TYPE_ABI, (scope, choice)) is None
        assert isinstance(require_class(scope, choice, book=book),
                          UnresolvedSourceDemand)
        before = tuple(book.clock)
        answer = require_class_alternatives(
            scope, choice, control_scope=fact.control_cell, book=book)
        assert isinstance(answer, ClassAlternativesEvidence)
        assert answer.alternatives == fact.alternatives
        assert set((fact.control_cell, fact.operand_cell,
                    *fact.arm_value_cells, *fact.arm_class_cells)) <= set(
                        answer.proof_cells)
        assert CorrelationTable._source_closed_class_alternative_findings(
            SimpleNamespace(metadata={"identity_book": book})) == []
        assert tuple(book.clock) == before
        wrong_control = require_class_alternatives(scope, choice, book=book)
        assert wrong_control.missing == "missing_exact_class_alternative_control"
    finally:
        end_identity_book(token)


def test_unproved_arm_leaves_class_domain_open():
    book, token = begin_identity_book()
    try:
        _function, scope, choice = _choice(
            CLASSES + "def choose(flag: bool, other):\n"
            "    return A() if flag else other\n")
        assert book.latest_ref(SOURCE_CLOSED_CLASS_ALTERNATIVES,
                               (scope, choice)) is None
        answer = require_class_alternatives(scope, choice, book=book)
        assert isinstance(answer, UnresolvedSourceDemand)
        assert answer.missing == "missing_closed_class_alternatives"
    finally:
        end_identity_book(token)
