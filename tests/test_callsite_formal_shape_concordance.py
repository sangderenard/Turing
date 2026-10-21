"""Callsite shape facts follow authored identities into specialized graphs."""

import ast
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.compiler.glsl_deployment_strategy import (
    _invalidate_tensor_descriptor_dependents,
    _publish_callsite_return_members,
    _proven_formal_shape,
    _publish_formal_shape,
    _tensor_descriptor,
)
from src.compiler.tensor_ssa_lowering import _settle_operand_shapes
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGEST_SOURCE,
    INGESTION,
)
from src.compiler.identity_concordance import (
    Mode,
    Novel,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
    proven_shape_of,
    record_proven_shape,
    shape_scope_of,
)
from src.transmogrifier.ssa import SSAValue
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_constant_array_descriptor_enters_the_value_concordance():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "pointer_lane"
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3,
        constant=np.zeros(96, dtype=np.float64),
        attributes={"structural_specialization": True},
        parents=[], children=[],
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 3) == {
            "shape": (96,), "dtype": "float64", "rank": 1,
        }
        assert proven_shape_of(shape_scope_of(graph), 3) == (96,)
    finally:
        end_identity_book(token)


def test_builtin_scalar_cast_declares_rank_zero_result_descriptor():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "scalar_cast_result"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "value", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="float", op="float", value_id=2,
        attributes={"extraction_identity": "builtins.float"},
        parents=[(1, "arg:0")], children=[],
    )
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3, constant=1.0,
        attributes={"value": 1.0}, parents=[], children=[],
    )
    graph.G.add_node(
        4, type="min", op="min", value_id=4,
        expr_obj=ast.parse("min(float(value), 1.0)", mode="eval").body,
        attributes={"extraction_identity": "builtins.min"},
        parents=[(2, "arg:0"), (3, "arg:1")], children=[],
    )
    graph.G.add_edge(1, 2)
    graph.G.add_edges_from(((2, 4), (3, 4)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 2) == {
            "shape": (), "dtype": "float64", "rank": 0,
        }
        assert _tensor_descriptor(graph, 4) == {
            "shape": (), "dtype": "float64", "rank": 0,
        }
    finally:
        end_identity_book(token)


def test_declared_scalar_record_field_is_a_rank_zero_descriptor():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "scalar_record_field"
    record = {
        "identity": "example.Targets",
        "fields": {
            "limit": {
                "storage": "scalar", "dtype": "float64", "rank": 0,
            },
        },
    }
    graph.G.graph["program_abi"] = {"records": {"Targets": record}}
    graph.G.graph["parameter_record_abi"] = {"targets": record}
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "targets", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="GetAttr", op="GetAttr", value_id=2,
        expr_obj=ast.parse("targets.limit", mode="eval").body,
        attributes={"attribute": "limit"},
        parents=[(1, "value")], children=[],
    )
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3, constant=2.0,
        attributes={"value": 2.0},
        parents=[], children=[],
    )
    graph.G.add_node(
        4, type="Div", op="Div", value_id=4,
        expr_obj=ast.parse("measure / targets.limit", mode="eval").body,
        attributes={}, parents=[(3, "lhs"), (2, "rhs")], children=[],
    )
    graph.G.add_edges_from(((1, 2), (3, 4), (2, 4)))

    _book, token = begin_identity_book()
    try:
        expected = {"shape": (), "dtype": "float64", "rank": 0}
        assert _tensor_descriptor(graph, 2) == expected
        assert _tensor_descriptor(graph, 4) == expected
    finally:
        end_identity_book(token)


def test_numeric_literal_views_carry_shape_into_matmul_concordance():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "literal_projection"
    graph.G.add_node(
        1, type="Constant", op="const", value_id=1,
        constant=[float(index) for index in range(8)],
        attributes={"value": [float(index) for index in range(8)]},
        parents=[],
    )
    for node_id, value in ((2, 0), (4, 2), (6, 0), (8, 1)):
        graph.G.add_node(
            node_id, type="Constant", op="const", value_id=node_id,
            constant=value, attributes={"value": value}, parents=[],
        )
    graph.G.add_node(
        3, type="unsqueeze", op="unsqueeze", value_id=3,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(1, "operand"), (2, "arg:0")],
    )
    graph.G.add_node(
        5, type="unsqueeze", op="unsqueeze", value_id=5,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(3, "operand"), (4, "arg:0")],
    )
    graph.G.add_node(
        7, type="unsqueeze", op="unsqueeze", value_id=7,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(1, "operand"), (6, "arg:0")],
    )
    graph.G.add_node(
        9, type="unsqueeze", op="unsqueeze", value_id=9,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(7, "operand"), (8, "arg:0")],
    )
    graph.G.add_node(
        10, type="matmul", op="matmul", value_id=10,
        attributes={}, parents=[(5, "lhs"), (9, "rhs")],
    )
    graph.G.add_edges_from((
        (1, 3), (2, 3), (3, 5), (4, 5),
        (1, 7), (6, 7), (7, 9), (8, 9), (5, 10), (9, 10),
    ))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 10) == {
            "shape": (1, 8, 8), "dtype": "float64", "rank": 3,
        }
        assert proven_shape_of(shape_scope_of(graph), 10) == (1, 8, 8)
    finally:
        end_identity_book(token)


def test_specialized_graph_reads_authored_formal_shape_row():
    """A specialization suffix must not create a second formal identity."""

    graph = nx.DiGraph()
    graph.graph["function_name"] = (
        "artifact___first_occurrence__specialized_5171c8115542"
    )
    specialized = SimpleNamespace(G=graph)

    _book, token = begin_identity_book()
    try:
        _publish_formal_shape(
            "_first_occurrence",
            "mask",
            {"shape": (2,), "dtype": "float64"},
            "_pivot_mask",
        )

        assert _proven_formal_shape(specialized, "mask") == {
            "shape": (2,),
            "dtype": "float64",
            "rank": 1,
        }
        page = current_identity_book().page("formal_shape")
        assert page.rows() == (("_first_occurrence", "mask"),)
    finally:
        end_identity_book(token)


def test_parametric_formal_shape_repeated_caller_is_not_a_new_mutation():
    """A settled conflict changes only when a genuinely new callsite is read."""

    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        for value_id, name in enumerate(("first", "second", "third"), 1):
            book.post(
                CANONICAL_VALUE, (("callers", 0), value_id),
                ("Input", "input", name),
                stage=INGESTION,
                provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )
        page = book.page("formal_shape")
        first = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 1),
        )
        second = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 2),
        )
        third = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 3),
        )

        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (2,), "dtype": "float64"}, "first",
            (first,),
        )
        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (3,), "dtype": "float64"}, "second",
            (second,),
        )
        conflict_history_length = len(page.history(("callee", "value")))

        assert not _publish_formal_shape(
            "callee", "value",
            {"shape": (3,), "dtype": "float64"}, "second",
            (second,),
        )
        assert len(page.history(("callee", "value"))) == conflict_history_length

        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (4,), "dtype": "float64"}, "third",
            (third,),
        )
        third_history_length = len(page.history(("callee", "value")))
        assert not _publish_formal_shape(
            "callee", "value",
            {"shape": (2,), "dtype": "float64"}, "first",
            (first,),
        )
        assert len(page.history(("callee", "value"))) == third_history_length
    finally:
        end_identity_book(token)


def test_exact_polymorphic_shape_conflicts_at_current_concordance_depth():
    """A later exact specialization must not inherit a peer's deeper shape."""

    _book, token = begin_identity_book()
    try:
        record_proven_shape("_row", 6, (2, 2), "float64", 4)
        record_proven_shape("_row", 6, (2, 1), "float64", None)

        assert proven_shape_of("_row", 6) is None
        page = current_identity_book().page("proven_shape")
        assert page.latest(("_row", 6))[0] == "conflicting"
    finally:
        end_identity_book(token)


def test_polymorphic_helper_keeps_exact_specialization_operand_shape():
    """A peer specialization's proven extent cannot restamp a local value."""

    _book, token = begin_identity_book()
    try:
        record_proven_shape("_row", 6, (2, 2), "float64", 2)
        page = current_identity_book().page("value_shape")
        page.set(
            ("_row", 0), 0,
            ("from matrix caller", (2, 2), "float64", "span"),
        )
        page.set(
            ("_row", 0), 1,
            ("from column caller", (2, 1), "float64", "span"),
        )
        local = SSAValue(6, dtype="float64", shape=(2, 1))

        column = SimpleNamespace(
            name="artifact___row__specialized_column", blocks={}, metadata={},
        )
        _settle_operand_shapes(column, (local,))

        assert local.shape == (2, 1)
    finally:
        end_identity_book(token)


def test_region_operand_reads_later_unanimous_tensor_descriptor_shape():
    """A captured tensor projection cannot remain scalar at its consumer."""

    _book, token = begin_identity_book()
    try:
        operand = SSAValue(710, dtype="float64", shape=())
        region = SimpleNamespace(
            name="artifact__planned_region_267", blocks={}, metadata={},
        )
        descriptor = SimpleNamespace(shape=(144, 288))

        _settle_operand_shapes(
            region, (operand,), lambda value: (
                descriptor if int(value.id) == 710 else None
            ),
        )

        assert operand.shape == (144, 288)
    finally:
        end_identity_book(token)


def test_existing_call_projection_is_enriched_by_later_exact_descriptor():
    """A resident projection id does not freeze an incomplete first round."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "solve"
    graph.G.add_node(
        27, type="Call", op="Call", value_id=27,
        attributes={}, parents=[], children=[],
    )
    initial = (
        {"shape": (2, 2), "dtype": "float64"},
        {"shape": (), "dtype": "float64"},
        None,
    )
    exact = (
        initial[0], initial[1],
        {"shape": (2, 2), "dtype": "float64"},
    )

    _book, token = begin_identity_book()
    try:
        assert _publish_callsite_return_members(
            graph, 27, initial, "tuple",
        )
        leaves = tuple(
            graph.G.nodes[27]["attributes"]["aggregate_leaf_value_ids"]
        )
        permutation = leaves[2]
        assert graph.G.nodes[permutation]["tensor"] == {}

        assert _publish_callsite_return_members(
            graph, 27, exact, "tuple",
        )

        assert tuple(
            graph.G.nodes[27]["attributes"]["aggregate_leaf_value_ids"]
        ) == leaves
        assert graph.G.nodes[permutation]["tensor"] == exact[2]
        # Descriptor revisions now live on the shared declared member page,
        # keyed by caller-copy scope, not the retired action-log name page.
        page = current_identity_book().page("callsite_return_member")
        history = page.history((("function", "solve"), 27, 2, permutation))
        assert history == (
            (0, None),
            (1, ((2, 2), "float64")),
        )
    finally:
        end_identity_book(token)


def test_polymorphic_specialization_rederives_intermediate_shape_locally():
    """A peer's authored-row proof cannot enter an exact descriptor graph."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "_row",
        "planner_tensor_descriptors": {
            "matrix": {"shape": (2, 1), "dtype": "float64"},
            "selector": {"shape": (2, 1), "dtype": "float64"},
        },
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (2, 1), "dtype": "float64"},
        attributes={"binding_name": "matrix"}, parents=[], children=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (2, 1), "dtype": "float64"},
        attributes={"binding_name": "selector"}, parents=[], children=[],
    )
    graph.G.add_node(
        2, type="Mul", op="mul", value_id=2,
        parents=[(0, "lhs"), (1, "rhs")], children=[(3, "operand")],
        attributes={"tensor_candidate": "mul"},
    )
    graph.G.add_node(
        3, type="sum", op="sum", value_id=3,
        parents=[(2, "operand")], children=[],
        attributes={"tensor_candidate": "sum", "dim": -2},
    )
    graph.G.add_edges_from(((0, 2), (1, 2), (2, 3)))

    _book, token = begin_identity_book()
    try:
        record_proven_shape(shape_scope_of(graph), 2, (2, 2), "float64", 2)
        _publish_formal_shape(
            "_row", "matrix",
            {"shape": (2, 2), "dtype": "float64"}, "matrix_lane",
        )
        _publish_formal_shape(
            "_row", "matrix",
            {"shape": (2, 1), "dtype": "float64"}, "column_lane",
        )

        assert _tensor_descriptor(graph, 3) == {
            "shape": (1,), "dtype": "float64", "rank": 1,
        }
        history = current_identity_book().page("proven_shape").history(
            (shape_scope_of(graph), 2)
        )
        assert history[-1][1][0] == "conflicting"
    finally:
        end_identity_book(token)


def test_axis_reduction_keeps_tracked_result_when_operand_rank_is_unsettled():
    """A transient scalar operand fact cannot erase a reduction descriptor."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "reduce_rows",
        "planner_tensor_descriptors": {"value": {
            "shape": (), "dtype": "float64",
        }},
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (), "dtype": "float64"},
        attributes={"binding_name": "value"}, parents=[], children=[
            (1, "operand"),
        ],
    )
    graph.G.add_node(
        1, type="sum", op="sum", value_id=1,
        tensor={"shape": (8,), "dtype": "float64"},
        parents=[(0, "operand")], children=[],
        attributes={"tensor_candidate": "sum", "dim": 2},
    )
    graph.G.add_edge(0, 1)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 1) == {
            "shape": (8,), "dtype": "float64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_leading_axis_reduction_uses_dynamic_sequence_row_shape():
    """A dynamic row count does not hide the statically known row extents."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "reduce_dynamic_rows"
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={
            "shape": (),
            "dtype": "float64",
            "rank": 2,
            "metadata_state": "dynamic",
            "sequence_row_shape": (3,),
        },
        attributes={"binding_name": "rows"}, parents=[], children=[
            (1, "operand"),
        ],
    )
    graph.G.add_node(
        1, type="mean", op="mean", value_id=1,
        parents=[(0, "operand")], children=[],
        attributes={"tensor_candidate": "mean", "dim": 0},
    )
    graph.G.add_edge(0, 1)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 1) == {
            "shape": (3,), "dtype": "float64", "rank": 1,
        }
        assert proven_shape_of(shape_scope_of(graph), 1) == (3,)
    finally:
        end_identity_book(token)


def test_dependency_change_invalidates_and_then_reproves_concordance_shape():
    """Clearing a graph cache also withdraws its stale identity-level proof."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "helper"
    graph.G.add_node(1, type="Call", op="Call", value_id=1, parents=[])
    graph.G.add_node(
        2, type="Add", op="add", value_id=2,
        parents=[(1, "lhs")], tensor={"shape": (2,), "dtype": "float64"},
    )
    graph.G.add_edge(1, 2)

    _book, token = begin_identity_book()
    try:
        record_proven_shape(shape_scope_of(graph), 2, (2,), "float64", 4)
        _invalidate_tensor_descriptor_dependents(
            graph, (1,), "call-result-specialization-changed",
        )

        assert "tensor" not in graph.G.nodes[2]
        assert proven_shape_of(shape_scope_of(graph), 2) is None
        assert current_identity_book().page("proven_shape").latest(
            (shape_scope_of(graph), 2)
        ) == (
            "invalidated", 1, "call-result-specialization-changed",
        )

        record_proven_shape(shape_scope_of(graph), 2, (1,), "float64", 2)
        assert proven_shape_of(shape_scope_of(graph), 2) == (1,)
    finally:
        end_identity_book(token)
