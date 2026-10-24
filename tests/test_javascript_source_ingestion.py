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
    # C pow(1, NaN) is 1; JavaScript 1 ** NaN is NaN.
    "/** @param {number} x @param {number} y */ function f(x, y) { return x ** y; }",
    # Implementation-approximated: V8 and libm differ in the last bit.
    "/** @param {number} x */ function f(x) { return Math.sin(x); }",
    "/** @param {number} x @param {number} y */ function f(x, y) { return Math.pow(x, y); }",
])
def test_oracle_findings_stay_loud_refusals(source):
    lowered = lower_javascript_source_to_ssa(source)
    assert not lowered.complete


@needs_node
@pytest.mark.parametrize(
    "path", sorted(CORPUS.glob("c*.js")), ids=lambda path: path.stem,
)
def test_javascript_corpus_matches_node_bit_for_bit(path):
    source, inputs = split_program(path.read_text(encoding="utf-8"))
    status = check_program(source, inputs)
    assert status == "PASS", status
