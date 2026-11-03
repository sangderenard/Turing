"""Closed tuple arms publish length evidence before a native layout exists."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    SOURCE_AGGREGATE_LAYOUT, SOURCE_CLOSED_AGGREGATE_LENGTHS,
)
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.source_evidence_resolver import (
    LengthEvidence,
    UnresolvedSourceDemand,
    require_length,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _tuple_choice(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    function = graph.function_table.entry("choose").graph.G
    choice = next(
        int(node_id) for node_id, data in function.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.IfExp)
    )
    return function, function.graph["source_aggregate_layout_scope"], choice


def test_equal_tuple_arms_prove_exact_length_with_both_member_trails():
    book, token = begin_identity_book()
    try:
        graph, scope, choice = _tuple_choice(
            "def choose(flag: bool, a: int, b: int, c: int):\n"
            "    return (a, b) if flag else (a, c)\n"
        )
        before = tuple(book.clock)
        answer = require_length(scope, choice, book=book)
        assert isinstance(answer, LengthEvidence)
        assert answer.length == 2
        layout = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            answer.subject_cell.row, answer.subject_cell.column,
        )
        assert layout.conditional_cell in answer.proof_cells
        assert len(layout.arm_layout_cells) == 2
        assert set(layout.arm_layout_cells) <= set(answer.proof_cells)
        assert len(layout.members) == 2
        assert all(member.source_cells for member in layout.members)
        assert answer.control_scope == layout.conditional_cell
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_unequal_tuple_arms_retain_closed_lengths_but_refuse_exact_layout():
    book, token = begin_identity_book()
    try:
        graph, scope, choice = _tuple_choice(
            "def choose(flag: bool, a: int, b: int, c: int, d: int):\n"
            "    return (a, b) if flag else (a, b, c, d)\n"
        )
        # The authored IfExp has both exact tuple operands. Their members
        # remain distinct; choosing one length would erase a control arm.
        parents = dict((role, parent) for parent, role in graph.nodes[
            choice
        ]["parents"])
        arm_layouts = tuple(book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (
            scope, parents[arm],
        )) for arm in ("body", "orelse"))
        assert all(arm_layouts)
        assert tuple(len(book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            cell.row, cell.column,
        ).members) for cell in arm_layouts) == (2, 4)
        before = tuple(book.clock)
        answer = require_length(scope, choice, book=book)
        assert isinstance(answer, UnresolvedSourceDemand)
        assert answer.missing == "closed_length_alternatives_require_native_layout"
        assert answer.alternatives == (2, 4)
        assert answer.subject_cell is not None
        domain = book.page(SOURCE_CLOSED_AGGREGATE_LENGTHS).cell(
            answer.subject_cell.row, answer.subject_cell.column,
        )
        assert set(domain.arm_layout_cells) <= set(answer.proof_cells)
        assert domain.conditional_cell in answer.proof_cells
        for canonical_arm, source_arm in zip(
            arm_layouts, domain.arm_layout_cells, strict=True,
        ):
            assert source_arm in {
                parent for parent, _stage in book.edges_into(canonical_arm)
            }
        assert book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (
            scope, choice,
        )) is None
        assert CorrelationTable._source_closed_aggregate_length_findings(
            SimpleNamespace(metadata={"identity_book": book})
        ) == []
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)
