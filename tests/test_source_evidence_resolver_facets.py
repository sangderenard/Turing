"""Published class and tuple layout evidence is consumed without inference."""

import ast
import contextlib
import io

from src.common.tensors.topological_reducer import (
    _source_numeric_scope,
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import SOURCE_AGGREGATE_LAYOUT
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.source_evidence_resolver import (
    ClassEvidence,
    LengthEvidence,
    MemberEvidence,
    UnresolvedSourceDemand,
    require_class,
    require_length,
    require_member,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _reduce(source):
    process = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        process.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(process)
    graph = process.function_table.entry("choose").graph.G
    scope = graph.graph["source_aggregate_layout_scope"]
    result_id = next(
        int(data.get("value_id", node_id))
        for node_id, data in graph.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.IfExp)
    )
    return graph, scope, result_id


def test_equal_tuple_arms_publish_exact_length_members_and_class():
    book, token = begin_identity_book()
    try:
        graph, scope, result_id = _reduce(
            "def choose(flag, first: int, second: int, third: int):\n"
            "    return (first, second) if flag else (first, third)\n"
        )
        before = tuple(book.clock)
        length = require_length(scope, result_id, book=book)
        member = require_member(scope, result_id, 1, book=book)
        assert isinstance(length, LengthEvidence), getattr(length, "missing", None)
        assert length.length == 2
        assert length.control_scope is not None
        assert len(book.page(SOURCE_AGGREGATE_LAYOUT).latest((
            scope, result_id,
        )).arm_layout_cells) == 2
        assert isinstance(member, MemberEvidence)
        assert member.control_scope == length.control_scope
        assert member.member_cell in length.proof_cells
        assert member.proof_cells
        first_id = next(
            int(data.get("value_id", node_id))
            for node_id, data in graph.nodes(data=True)
            if (data.get("attributes") or {}).get("binding_name") == "first"
            and str(data.get("type")) == "Input"
        )
        klass = require_class(_source_numeric_scope(graph), first_id, book=book)
        assert isinstance(klass, ClassEvidence)
        assert klass.class_identity == "builtins.int"
        assert klass.proof_cells
        assert klass.control_scope == _source_numeric_scope(graph)
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)


def test_unequal_tuple_arms_keep_closed_lengths_without_merge_layout():
    book, token = begin_identity_book()
    try:
        _graph, scope, result_id = _reduce(
            "def choose(flag, first, second, third, fourth):\n"
            "    return (first, second) if flag else (first, second, third, fourth)\n"
        )
        before = tuple(book.clock)
        length = require_length(scope, result_id, book=book)
        member = require_member(scope, result_id, 0, book=book)
        assert isinstance(length, UnresolvedSourceDemand)
        assert length.missing == "closed_length_alternatives_require_native_layout"
        assert length.alternatives == (2, 4)
        assert length.proof_cells
        assert isinstance(member, UnresolvedSourceDemand)
        assert member.missing == "missing_aggregate_layout"
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)
