"""JavaScript source -> repository SSA -> native, checked against node."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from javascript_corpus_typed_check import (  # noqa: E402
    check_program,
    node_executable,
    split_program,
)
from src.compiler.javascript_source_ingestion import (  # noqa: E402
    lower_javascript_source_to_ssa,
)


CORPUS = ROOT / "tests" / "javascript_semantics_corpus"
needs_node = pytest.mark.skipif(
    node_executable() is None, reason="node is required for the JS parser/oracle",
)


@needs_node
def test_function_lowers_to_a_module_with_declared_abi():
    lowered = lower_javascript_source_to_ssa(
        "/** @param {number} a @param {boolean} p */\n"
        "function f(a, p) { return p ? a / 2 : -a; }"
    )
    assert lowered.complete, lowered.shortfall_report()
    function = lowered.module.functions["f"]
    assert [value.dtype for value in function.args] == ["float64", "bool"]
    assert dict(function.metadata["parameter_names"]) == {
        "a": function.args[0].id, "p": function.args[1].id,
    }
    constants = [
        instruction.attributes["value"]
        for instruction in function.blocks["entry"].instrs
        if instruction.op == "Const"
    ]
    # A JavaScript Number literal is a double, never an int64.
    assert constants == [2.0] and type(constants[0]) is float
    assert lowered.result_kind == "number"


@needs_node
def test_parameter_without_declared_kind_is_a_named_shortfall():
    lowered = lower_javascript_source_to_ssa("function f(a) { return a + 1; }")
    assert not lowered.complete
    assert any(
        item.code == "undeclared-parameter-kind" for item in lowered.shortfalls
    )


@needs_node
def test_jsdoc_of_another_function_does_not_type_this_one():
    lowered = lower_javascript_source_to_ssa(
        "/** @param {number} a */ function g(a) { return a; }\n"
        "function f(a) { return a; }",
        function_name="f",
    )
    assert any(
        item.code == "undeclared-parameter-kind" for item in lowered.shortfalls
    )


@needs_node
@pytest.mark.parametrize("source", [
    # JavaScript returns one of the Numbers; LAnd would return a bool.
    "/** @param {number} x @param {number} y */ function f(x, y) { return x && y; }",
    "/** @param {number} x @param {number} y */ function f(x, y) { return x || y; }",
])
def test_number_logical_operators_stay_loud_refusals(source):
    lowered = lower_javascript_source_to_ssa(source)
    assert not lowered.complete


@needs_node
@pytest.mark.parametrize("source", [
    # C pow(1, NaN) and pow(-1, Infinity) are 1; ECMAScript says NaN.
    "/** @param {number} a @param {number} b */ function root(a, b) { return a ** b; }",
    "/** @param {number} a @param {number} b */ function root(a, b) { return Math.pow(a, b); }",
])
def test_exponentiation_special_cases_match_node(source):
    rows = "[[1, NaN], [-1, Infinity], [1, -Infinity], [NaN, 0], [NaN, -0], [-8, 1/3], [0, -1], [-0, -1], [2, 0.5], [-2, 3]]"
    status = check_program(source, rows)
    assert status.startswith("PASS"), status


@needs_node
def test_transcendentals_compile_to_repository_operations_within_the_ulp_tier():
    source = (
        "/** @param {number} x */ function root(x) "
        "{ return Math.sin(x) + Math.exp(x) - Math.log(x + 2) + Math.atan(x); }"
    )
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    operations = {
        instruction.op
        for instruction in lowered.module.functions["root"].blocks["entry"].instrs
    }
    assert {"sin", "exp", "log", "atan"} <= operations
    assert "Call" not in operations
    status = check_program(source, "[[0.5], [1], [2.5], [-1.5], [10], [-0], [NaN], [Infinity]]")
    assert status.startswith("PASS (approximated"), status


@needs_node
@pytest.mark.parametrize(
    "path", sorted(CORPUS.glob("c*.js")), ids=lambda path: path.stem,
)
def test_javascript_corpus_matches_node(path):
    # Bit for bit, or within the ulp tier when the program uses an
    # implementation-approximated operation (the status then says so).
    source, inputs = split_program(path.read_text(encoding="utf-8"))
    status = check_program(source, inputs)
    assert status == "PASS" or status.startswith("PASS (approximated"), status


@needs_node
def test_sibling_scopes_may_declare_the_same_name():
    # Each arm's ``let t`` and each block's ``let u`` is its own binding;
    # the second once re-posted version 0 and the identity book refused it.
    source = (
        "/** @param {boolean} p */ function root(p) { let s = 0; "
        "if (p) { let t = 1; s = t; } else { let t = 2; s = t; } "
        "{ let u = 3; s += u; } { let u = 4; s += u; } return s; }"
    )
    assert check_program(source, "[[true], [false]]") == "PASS"


@needs_node
def test_loop_lowers_to_header_body_exit_blocks_with_phis():
    lowered = lower_javascript_source_to_ssa(
        "/** @param {number} n */ function root(n) "
        "{ let s = 0; for (let i = 0; i < n; i++) { s += i; } return s; }"
    )
    assert lowered.complete, lowered.shortfall_report()
    blocks = lowered.module.functions["root"].blocks
    assert list(blocks) == ["entry", "loop0_header", "loop0_body", "loop0_exit"]
    phis = [i for i in blocks["loop0_header"].instrs if i.op == "Phi"]
    assert sorted(i.attributes["binding_name"] for i in phis) == ["i", "s"]
    assert all(
        i.attributes["incoming_blocks"] == ("entry", "loop0_body") for i in phis
    )


@needs_node
@pytest.mark.parametrize("source", [
    # break/continue/return leave a loop other than through its condition.
    "/** @param {number} n */ function root(n) { let s = 0; "
    "for (let i = 0; i < n; i++) { if (i > 2) { break; } s += i; } return s; }",
    "/** @param {number} n */ function root(n) { let s = 0; "
    "while (s < n) { return s; } return 1; }",
    # A carried value that changes kind has no single dtype.
    "/** @param {number} n */ function root(n) { let x = 0; "
    "for (let i = 0; i < n; i++) { x = x > 0; } return x; }",
])
def test_loop_exits_and_kind_changes_stay_loud_refusals(source):
    assert not lower_javascript_source_to_ssa(source).complete


@needs_node
def test_sibling_loops_may_declare_the_same_counter():
    # Two loops' ``let i`` and two blocks' ``let t`` are distinct bindings;
    # the second once collided with the first on the identity book.
    source = (
        "/** @param {number} n */ function root(n) { let s = 0; "
        "for (let i = 0; i < n; i++) { s += i; } "
        "for (let i = 0; i < 2; i++) { s *= 2; } "
        "{ let t = 1; s += t; } { let t = 2; s += t; } return s; }"
    )
    assert check_program(source, "[[0], [3], [NaN]]") == "PASS"


SPECIAL_PAIRS = (
    "[[5, 3], [-5, 3], [5, -3], [-0, 3], [0, -3], [5, 0], [NaN, 1], [1, NaN], "
    "[Infinity, 2], [2, Infinity], [-2, -Infinity], [-0, -0], [0, -0], [-0, 0]]"
)


@needs_node
@pytest.mark.parametrize("expression", [
    # Truncated remainder with the dividend's sign; Mod alone is floored.
    "a % b",
    # NaN wins and equal zeros order +0 above -0; a ternary or fmax does not.
    "Math.max(a, b)", "Math.min(a, b)", "Math.min(b, a)",
    # +-0 and NaN come back unchanged.
    "Math.sign(a)",
])
def test_remainder_and_math_extrema_are_exact_against_node(expression):
    source = (
        "/** @param {number} a @param {number} b */ "
        f"function root(a, b) {{ return {expression}; }}"
    )
    assert check_program(source, SPECIAL_PAIRS) == "PASS"


@needs_node
@pytest.mark.parametrize("source", [
    "/** @param {boolean} p @param {number} b */ function root(p, b) { return p % b; }",
    "/** @param {number} a */ function root(a) { return Math.max(a, a, 1); }",
])
def test_remainder_of_a_boolean_and_variadic_max_stay_loud_refusals(source):
    assert not lower_javascript_source_to_ssa(source).complete
