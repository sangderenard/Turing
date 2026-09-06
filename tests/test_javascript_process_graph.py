from __future__ import annotations

import pytest

from src.compiler.dream_document import _run_javascript_ast_parser
from src.compiler.javascript_process_graph import (
    ingest_javascript_expression,
    javascript_source_from_graph,
)
from src.compiler.javascript_source_tables import (
    JAVASCRIPT_BINARY_TO_SSA,
    JAVASCRIPT_UNARY_TO_SSA,
    JAVASCRIPT_UNSUPPORTED_BINARY,
    SSA_TO_JAVASCRIPT_BINARY,
    SSA_TO_JAVASCRIPT_UNARY,
    validate_invertible_tables,
)
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa_registry import Handler


def _parse_expression(source: str) -> dict:
    program = _run_javascript_ast_parser(f"const _turing_probe = {source};")
    statement = program["body"][0]
    return statement["declarations"][0]["init"]


def test_operator_tables_are_invertible_and_share_the_ssa_handler_vocabulary():
    validate_invertible_tables()
    assert JAVASCRIPT_BINARY_TO_SSA["+"] is Handler.Add
    assert JAVASCRIPT_BINARY_TO_SSA["*"] is Handler.Mul
    assert JAVASCRIPT_UNARY_TO_SSA["-"] is Handler.Neg
    # Same Handler names GLSL and sympy already lower into -- not a parallel
    # JS-only vocabulary.
    from src.compiler.glsl_source_tables import GLSL_BINARY_TO_SSA
    assert JAVASCRIPT_BINARY_TO_SSA["+"] is GLSL_BINARY_TO_SSA["+"]
    assert JAVASCRIPT_BINARY_TO_SSA["*"] is GLSL_BINARY_TO_SSA["*"]


@pytest.mark.parametrize("source", [
    "a + b",
    "a - b * c",
    "(a + b) * (c - d)",
    "-a",
    "!flag",
    "a === b",
    "a && b || c",
    "a ? b : c",
])
def test_from_javascript_produces_canonical_handler_nodes(source):
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_expression(graph, expression, owner="probe")

    assert lowering.complete, lowering.shortfalls
    root_type = graph.G.nodes[lowering.root]["type"]
    assert root_type in {handler.name for handler in Handler}


@pytest.mark.parametrize("source", [
    "a + b",
    "a - b * c",
    "(a + b) * (c - d)",
    "-a",
    "!flag",
    "a === b",
    "Math.sqrt(a)",
    "a ? b : c",
])
def test_from_javascript_lowers_through_the_real_aot_ssa_pipeline(source):
    # Regression guard: an earlier version of ingest_javascript_expression
    # hand-rolled graph.G.add_node()/add_edge() with made-up attribute names
    # and reported these same shortfall-free/round-trip results without ever
    # running the output through the compiler. It failed immediately with
    # KeyError: 'children' the first time it was actually asked to schedule.
    # This test is the thing that would have caught that.
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_expression(graph, expression, owner="probe")
    assert lowering.complete, lowering.shortfalls

    graph.scheduler = type(graph.scheduler)(graph)
    instrs = process_graph_to_ssa_instrs(graph, schedule="asap")
    assert len(instrs) > 0


@pytest.mark.parametrize("source", [
    "a + b",
    "a - b * c",
    "(a + b) * (c - d)",
    "-a",
    "!flag",
    "a === b",
    "a && b",
    "Math.sqrt(a)",
    "1",
    "true",
    "a ? b : c",
])
def test_javascript_round_trips_through_the_process_graph(source):
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_expression(graph, expression, owner="probe")
    assert lowering.complete, lowering.shortfalls

    rendered = javascript_source_from_graph(graph, lowering.root)

    # Re-parse and re-ingest the rendered text; the second ingestion must
    # reach an identical Handler/topology shape -- proving the round trip
    # is semantically stable, not just that some string came out.
    reparsed = _parse_expression(rendered)
    graph_2 = ProcessGraph(materialize_memory=False)
    lowering_2 = ingest_javascript_expression(graph_2, reparsed, owner="probe")
    assert lowering_2.complete, lowering_2.shortfalls
    assert (
        graph.G.nodes[lowering.root]["type"]
        == graph_2.G.nodes[lowering_2.root]["type"]
    )


@pytest.mark.parametrize("source,code", [
    ("a == b", "unsupported-binary-operator"),
    ("a >>> b", "unsupported-binary-operator"),
    ("typeof a", "unsupported-unary-operator"),
    ("[a, b]", "unsupported-node-type"),
    ("({a: b})", "unsupported-node-type"),
])
def test_unsupported_operators_are_reported_not_silently_dropped(source, code):
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_expression(graph, expression, owner="probe")

    assert not lowering.complete
    assert any(item.code == code for item in lowering.shortfalls)


def test_reverse_tables_are_exact_inverses_of_the_forward_tables():
    for spelling, handler in JAVASCRIPT_BINARY_TO_SSA.items():
        assert SSA_TO_JAVASCRIPT_BINARY[handler] == spelling
    for spelling, handler in JAVASCRIPT_UNARY_TO_SSA.items():
        assert SSA_TO_JAVASCRIPT_UNARY[handler] == spelling
    assert "==" in JAVASCRIPT_UNSUPPORTED_BINARY
