"""Callsite shape facts follow authored identities into specialized graphs."""

from types import SimpleNamespace

import networkx as nx

from src.compiler.glsl_deployment_strategy import (
    _invalidate_tensor_descriptor_dependents,
    _publish_callsite_return_members,
    _proven_formal_shape,
    _publish_formal_shape,
    _tensor_descriptor,
)
from src.compiler.tensor_ssa_lowering import _settle_operand_shapes
from src.compiler.identity_concordance import (
    begin_identity_book,
    current_identity_book,
    end_identity_book,
    proven_shape_of,
    record_proven_shape,
)
from src.transmogrifier.ssa import SSAValue
from src.transmogrifier.graph.graph_express2 import ProcessGraph


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

        _settle_operand_shapes(
            "artifact___row__specialized_column", (local,),
        )

        assert local.shape == (2, 1)
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
        page = current_identity_book().page(
            "callsite_projection_specialization"
        )
        history = page.history(("solve", 27, 2, permutation))
        assert history == (
            (0, ("materialized", None)),
            (1, ("enriched", ((2, 2), "float64"))),
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
        record_proven_shape("_row", 2, (2, 2), "float64", 2)
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
            ("_row", 2)
        )
        assert history[-1][1][0] == "conflicting"
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
        record_proven_shape("helper", 2, (2,), "float64", 4)
        _invalidate_tensor_descriptor_dependents(
            graph, (1,), "call-result-specialization-changed",
        )

        assert "tensor" not in graph.G.nodes[2]
        assert proven_shape_of("helper", 2) is None
        assert current_identity_book().page("proven_shape").latest(
            ("helper", 2)
        ) == (
            "invalidated", 1, "call-result-specialization-changed",
        )

        record_proven_shape("helper", 2, (1,), "float64", 2)
        assert proven_shape_of("helper", 2) == (1,)
    finally:
        end_identity_book(token)
