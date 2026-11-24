"""A backward helper whose formal's value is its result type.

``backward_registry.unbroadcast(G, target_shape)``: the length of
``target_shape`` is the result rank and its entries decide which axes are
summed, so the helper has no single fixed-rank signature.  The registry
declares it (``HELPER_STATIC_SHAPE_PARAMETERS``); the planner identifies the
helper by its ``backward_rule_definition`` declaration, posts the
``static_shape_parameter`` row from it, and refuses a callsite that cannot
supply the formal as a literal.
"""

from types import SimpleNamespace

import pytest

from src.common.tensors import backward_registry
from src.compiler import glsl_deployment_strategy as planner
from src.compiler.concordance_declarations import (
    BACKWARD_RULE_DEFINITION, SOURCE_SPAN, STATIC_SHAPE_PARAMETER,
)
from src.compiler.identity_concordance import current_identity_book
from src.compiler.process_graph_autograd import _compiled_backward_rule_process_graph


def _rule_graph_entries():
    graph = _compiled_backward_rule_process_graph()
    entries = {entry.name: entry for entry in graph.function_table}
    return graph, entries


def test_generated_helper_definition_derives_from_its_declaration():
    _graph, entries = _rule_graph_entries()
    identity = entries["unbroadcast"].graph.G.graph["authored_function_identity"]
    book = current_identity_book()
    span = book.latest_ref(SOURCE_SPAN, (*identity, ()))

    assert span is not None
    assert [
        (source.page, tuple(source.row)) for source, _stage in book.edges_into(span)
    ] == [(
        BACKWARD_RULE_DEFINITION,
        (backward_registry.__name__, "unbroadcast"),
    )]


def test_unbroadcast_declares_target_shape_static():
    _graph, entries = _rule_graph_entries()
    unbroadcast = entries["unbroadcast"].graph

    declared, declaration_cell = planner._declared_backward_helper(unbroadcast)
    assert declared == (backward_registry.__name__, "unbroadcast")
    assert declaration_cell.page == BACKWARD_RULE_DEFINITION
    assert planner._static_shape_parameters(unbroadcast) == ("target_shape",)
    assert planner._static_shape_parameters(entries["expand_to"].graph) == ()
    # A rule is not a registry helper.
    assert planner._declared_backward_helper(entries["bw_mul"].graph) is None

    book = current_identity_book()
    row = (backward_registry.__name__, "unbroadcast", "target_shape")
    cell = book.latest_ref(STATIC_SHAPE_PARAMETER, row)
    assert cell is not None and book.page(STATIC_SHAPE_PARAMETER).latest(row) is True
    assert [source for source, _stage in book.edges_into(cell)] == [declaration_cell]


def test_callsite_without_a_literal_shape_is_refused():
    graph, entries = _rule_graph_entries()
    unbroadcast = entries["unbroadcast"].graph
    caller = entries["bw_mul"].graph

    planner._require_static_shape_literals(
        caller, 5, unbroadcast, {"target_shape": (1, 1)},
    )
    with pytest.raises(ValueError, match="formal 'target_shape' called from 'bw_mul' node 5"):
        planner._require_static_shape_literals(caller, 5, unbroadcast, {})


def test_callsite_left_generic_is_refused(monkeypatch):
    graph, _entries = _rule_graph_entries()
    reference = int(graph.function_table.reference("unbroadcast").address)
    generic = type("GenericShell", (), {})
    monkeypatch.setattr(
        planner, "_callsite_literal_shell_type",
        lambda owner, node_id, ref, fallback, limit: fallback,
    )

    with pytest.raises(ValueError, match="static-shape parameter has no literal"):
        planner._callsite_specialized_shell_type(
            SimpleNamespace(process_graph=graph), 5, reference, generic, 64,
        )
    # A helper with no declared static-shape formal keeps its generic shell.
    expand_to = int(graph.function_table.reference("expand_to").address)
    assert planner._callsite_specialized_shell_type(
        SimpleNamespace(process_graph=graph), 5, expand_to, generic, 64,
    ) is generic
