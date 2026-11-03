"""Structural len reads the reducer's exact conditional tuple layout."""

import ast

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    PROVEN_LITERAL, SOURCE_PYTHON_IDENTITY,
)
from src.compiler.glsl_deployment_strategy import _fold_callsite_structural_values
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.source_evidence_resolver import (
    LengthEvidence, UnresolvedSourceDemand, require_length,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _choice(else_arm):
    source = (
        "def choose(flag: bool, a: int, b: int, c: int, d: int):\n"
        f"    value = (a, b) if flag else {else_arm}\n"
        "    return len(value)\n"
    )
    process = ProcessGraph(materialize_memory=False)
    process.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(process)
    graph = process.function_table.entry("choose").graph
    choice = next(
        int(node_id) for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.IfExp)
    )
    call = next(
        int(node_id) for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    return graph, choice, call


def test_equal_conditional_tuple_len_folds_from_exact_layout():
    book, token = begin_identity_book()
    try:
        graph, choice, call = _choice("(a, c)")
        scope = graph.G.graph["source_aggregate_layout_scope"]
        length = require_length(scope, choice, book=book)
        assert isinstance(length, LengthEvidence)
        identity_row = graph.G.nodes[call]["attributes"]["source_python_identity_row"]
        identity_cell = book.latest_ref(SOURCE_PYTHON_IDENTITY, identity_row)
        assert book.page(SOURCE_PYTHON_IDENTITY).cell(
            identity_cell.row, identity_cell.column,
        )[0] == "builtins.len"

        _fold_callsite_structural_values(graph)

        assert graph.G.nodes[call]["type"] == "Constant"
        assert graph.G.nodes[call]["constant"] == 2
        literal = book.latest_ref(PROVEN_LITERAL, ("choose", call))
        assert book.page(PROVEN_LITERAL).cell(literal.row, literal.column) == 2
        causes = {cell for cell, _stage in book.edges_into(literal)}
        assert identity_cell in causes
        assert length.subject_cell in causes
        assert set(length.proof_cells) <= causes
    finally:
        end_identity_book(token)


def test_unequal_conditional_tuple_len_keeps_named_unresolved_domain():
    book, token = begin_identity_book()
    try:
        graph, choice, call = _choice("(a, b, c, d)")
        scope = graph.G.graph["source_aggregate_layout_scope"]
        length = require_length(scope, choice, book=book)
        assert isinstance(length, UnresolvedSourceDemand)
        assert length.missing == "closed_length_alternatives_require_native_layout"
        assert length.alternatives == (2, 4)
        # Even a legacy intrinsic hint cannot choose one conditional arm.
        graph.G.nodes[call]["attributes"]["extraction_identity"] = "builtins.len"

        _fold_callsite_structural_values(graph)

        assert graph.G.nodes[call]["type"] == "Call"
        assert book.latest_ref(PROVEN_LITERAL, ("choose", call)) is None
    finally:
        end_identity_book(token)
