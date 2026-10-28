from __future__ import annotations

import json
import subprocess

import pytest

from src.compiler.dream_document import _run_javascript_ast_parser
from src.compiler.javascript_process_graph import (
    PORTABLE_NUMERIC_ASSUMPTIONS,
    ingest_javascript_expression,
    ingest_javascript_program,
    ingest_javascript_source,
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
from src.transmogrifier.graph.javascript_identity_programs import (
    JAVASCRIPT_IDENTITY_PROGRAMS,
    resolve_javascript_identity,
)
from src.transmogrifier.ssa_registry import Handler


def _parse_expression(source: str) -> dict:
    program = _run_javascript_ast_parser(f"const _turing_probe = {source};")
    statement = program["body"][0]
    return statement["declarations"][0]["init"]


def _ingest_expression(graph, expression):
    return ingest_javascript_expression(
        graph, expression, owner="probe",
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )


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


def test_javascript_identity_catalog_distinguishes_operators_objects_and_hosts():
    assert JAVASCRIPT_IDENTITY_PROGRAMS
    sqrt = resolve_javascript_identity("javascript.realm.Math.sqrt")
    append = resolve_javascript_identity("javascript.Array.prototype.push")
    worker = resolve_javascript_identity("javascript.host.Worker")

    assert sqrt.kind == "operator" and sqrt.operator == "sqrt"
    assert append.kind == "object" and append.effects == ("write:receiver",)
    assert worker.kind == "host" and worker.attributes["region"] == "browser-javascript"


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
    lowering = _ingest_expression(graph, expression)

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
    lowering = _ingest_expression(graph, expression)
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
    lowering = _ingest_expression(graph, expression)
    assert lowering.complete, lowering.shortfalls

    rendered = javascript_source_from_graph(graph, lowering.root)

    # Re-parse and re-ingest the rendered text; the second ingestion must
    # reach an identical Handler/topology shape -- proving the round trip
    # is semantically stable, not just that some string came out.
    reparsed = _parse_expression(rendered)
    graph_2 = ProcessGraph(materialize_memory=False)
    lowering_2 = _ingest_expression(graph_2, reparsed)
    assert lowering_2.complete, lowering_2.shortfalls
    assert (
        graph.G.nodes[lowering.root]["type"]
        == graph_2.G.nodes[lowering_2.root]["type"]
    )


@pytest.mark.parametrize("source,code", [
    ("a == b", "unsupported-binary-operator"),
    ("a & b", "unsupported-binary-operator"),
    ("a >>> b", "unsupported-binary-operator"),
    ("typeof a", "unsupported-unary-operator"),
    ("[a, b]", "unsupported-node-type"),
    ("({a: b})", "unsupported-node-type"),
])
def test_unsupported_operators_are_reported_not_silently_dropped(source, code):
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, expression)

    assert not lowering.complete
    assert any(item.code == code for item in lowering.shortfalls)


def test_reverse_tables_are_exact_inverses_of_the_forward_tables():
    for spelling, handler in JAVASCRIPT_BINARY_TO_SSA.items():
        assert SSA_TO_JAVASCRIPT_BINARY[handler] == spelling
    for spelling, handler in JAVASCRIPT_UNARY_TO_SSA.items():
        assert SSA_TO_JAVASCRIPT_UNARY[handler] == spelling
    assert "==" in JAVASCRIPT_UNSUPPORTED_BINARY


def test_untyped_javascript_does_not_claim_numeric_coercions_are_exact():
    expression = _parse_expression("a + b")
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_expression(graph, expression, owner="untyped")

    assert not lowering.complete
    assert any(
        item.code == "unproven-javascript-semantics"
        for item in lowering.shortfalls
    )


def test_archive_map_value_function_is_one_identity_grounded_graph_region():
    source = """
    const mapValue = (value, min, max) => {
      const returnVal = (value - min) / (max - min);
      if (!returnVal) { return 0; }
      return returnVal;
    };
    """
    program = _run_javascript_ast_parser(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_program(
        graph, program, owner="archive/Functions.js", entry="mapValue",
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )

    assert lowering.complete, lowering.shortfalls
    assert lowering.entry_identity.startswith(
        "archive/Functions.js#javascript-function:mapValue:"
    )
    records = list(lowering.binding_identities)
    assert [record["name"] for record in records[:4]] == [
        "value", "min", "max", "returnVal",
    ]
    assert len({record["declaration_identity"] for record in records[:4]}) == 4

    root = graph.G.nodes[lowering.root]
    assert root["type"] == Handler.Ret.name
    returned = root["parents"][0][0]
    assert graph.G.nodes[returned]["type"] == Handler.Select.name
    operations = {str(data.get("op")) for _, data in graph.G.nodes(data=True)}
    # Number ``!`` is a real JavaScript truthiness program: x is false for
    # both zero and NaN.  It must not collapse to repository LNot(NaN).
    assert {
        "Load", "Const", "Sub", "Div", "Eq", "Call", "LOr", "Select", "Ret",
    } <= operations

    graph.scheduler = type(graph.scheduler)(graph)
    instrs = process_graph_to_ssa_instrs(graph, schedule="asap")
    assert [instr.op for instr in instrs][-1] == Handler.Ret.name
    select = next(instr for instr in instrs if instr.op == Handler.Select.name)
    assert select.arg_roles == ["condition", "if_true", "if_false"]

    from src.common.tensors.topological_reducer import node_identity_cell
    assert all(node_identity_cell(graph, node_id) is not None for node_id in graph.G)
    from src.compiler.concordance_declarations import IDENTITY_TRANSITION
    from src.compiler.identity_concordance import current_identity_book
    scope = tuple(graph.G.graph["ingestion_value_scope"])
    transitions = current_identity_book().page(
        IDENTITY_TRANSITION
    ).scope_rows(scope)
    assert len(transitions) == sum(
        len(data.get("parents") or ()) for _node, data in graph.G.nodes(data=True)
    )

    projected = javascript_source_from_graph(graph, returned)
    oracle = subprocess.run(
        ["node", "-e", source + f"""
        const projected = (value, min, max) => {projected};
        const cases = [
          [0, 0, 10], [5, 0, 10], [-1, 0, 10], [10, 0, 10],
          [0, 0, 0]
        ];
        console.log(JSON.stringify(cases.map(args => [
          mapValue(...args), projected(...args)
        ])));
        """],
        check=True, capture_output=True, text=True,
    )
    assert all(
        authored == translated
        for authored, translated in json.loads(oracle.stdout)
    )


def test_grounded_math_call_keeps_realm_identity_on_the_graph_node():
    expression = _parse_expression("Math.sqrt(a)")
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, expression)

    assert lowering.complete, lowering.shortfalls
    call = graph.G.nodes[lowering.root]
    assert call["attributes"]["javascript_identity"] == (
        "javascript.realm.Math.sqrt"
    )
    assert call["attributes"]["callee"] == "sqrt"


def test_reassigned_let_binding_is_a_new_version_of_the_same_name():
    program = _run_javascript_ast_parser("""
    function bump(x) {
      let y = x;
      y = y + 1;
      return y;
    }
    """)
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_program(
        graph, program, owner="assignment.js", entry="bump",
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )

    assert lowering.complete, lowering.shortfalls
    versions = [
        (record["name"], record["version"])
        for record in lowering.binding_identities
        if record["name"] == "y"
    ]
    assert versions == [("y", 0), ("y", 1)]


@pytest.mark.parametrize("source", [
    # A var read before its declaration runs reads undefined (Lane J12
    # admits only the vars that read like a let).
    "function f(x) { let z = y; var y = x; return z; }",
    "function f(x) { const y = x; y = 1; return y; }",
    "function f(x) { let y = x; y &&= 1; return y; }",
])
def test_var_const_reassignment_and_conditional_assignment_are_refused(source):
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_program(
        graph, _run_javascript_ast_parser(source), owner="assignment.js",
        entry="f", assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )
    assert not lowering.complete


def test_shadowed_math_is_not_resolved_as_a_realm_identity():
    program = _run_javascript_ast_parser(
        "function f(Math, a) { return Math.sqrt(a); }"
    )
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_program(
        graph, program, owner="shadow.js", entry="f",
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )

    assert not lowering.complete
    assert any(item.code == "shadowed-javascript-realm" for item in lowering.shortfalls)


@pytest.mark.parametrize("source", ["Math.sqrt()", "Math.sqrt(a, b)", "Math.max(a, b, c)"])
def test_math_identity_arity_is_exact(source):
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, _parse_expression(source))

    assert not lowering.complete
    assert any(item.code == "unsupported-call-arity" for item in lowering.shortfalls)


def test_math_call_uses_the_existing_tensor_call_lowering_contract():
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, _parse_expression("Math.sqrt(a)"))

    assert lowering.complete, lowering.shortfalls
    attributes = graph.G.nodes[lowering.root]["attributes"]
    assert attributes["tensor_operation"] == "sqrt"


def test_real_archive_nested_map_value_is_discovered_when_archive_is_present():
    from pathlib import Path
    import zipfile

    archive = Path(__file__).resolve().parents[2] / "archives" / "fit-datastudio-js-2023.zip"
    if not archive.exists():
        pytest.skip("workspace JavaScript archive is not present")
    with zipfile.ZipFile(archive) as bundle:
        source = bundle.read("eclipse-workspace/DataProcessor/Functions.js").decode()
    graph = ProcessGraph(materialize_memory=False)
    lowering = ingest_javascript_source(
        graph, source, owner=str(archive), entry="mapValue",
        assumptions=PORTABLE_NUMERIC_ASSUMPTIONS,
    )

    assert lowering.complete, lowering.shortfalls
    assert graph.G.nodes[lowering.root]["op"] == Handler.Ret.name


def test_math_round_remains_a_named_shortfall_not_repository_round_even():
    expression = _parse_expression("Math.round(a)")
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, expression)

    assert not lowering.complete
    assert any(item.code == "unsupported-call" for item in lowering.shortfalls)
    assert all(data.get("op") != "round" for _, data in graph.G.nodes(data=True))


@pytest.mark.parametrize("source", ["1 / 2", "2 ** 64", "!(n + 1)", "x ? 1 : 2"])
def test_number_literals_are_doubles_not_python_ints(source):
    # Acorn's JSON gives ``1`` as a Python int, which repository scalar
    # typing reads as int64.  Every JavaScript Number is a double, so no
    # numeric Constant may reach lowering as an int.
    expression = _parse_expression(source)
    graph = ProcessGraph(materialize_memory=False)
    lowering = _ingest_expression(graph, expression)
    assert lowering.complete, lowering.shortfalls

    constants = [
        data for _, data in graph.G.nodes(data=True)
        if data.get("type") == "Constant"
    ]
    assert constants
    for data in constants:
        if isinstance(data["constant"], bool):
            continue
        assert type(data["constant"]) is float
        assert type(data["attributes"]["value"]) is float
