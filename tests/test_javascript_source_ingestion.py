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
@pytest.mark.parametrize("expression", ["x && y", "x || y", "(x && y) || 7"])
def test_number_logical_operators_return_an_operand_like_node(expression):
    # JavaScript returns one of the Numbers (``-0 && 5`` is -0, ``NaN || 0``
    # is 0); LAnd/LOr would return a bool.
    source = (
        "/** @param {number} x @param {number} y */ "
        f"function root(x, y) {{ return {expression}; }}"
    )
    rows = "[[5, 3], [0, 3], [-0, 3], [NaN, 1], [1, NaN], [0, -0], [NaN, 0]]"
    assert check_program(source, rows) == "PASS"


@needs_node
def test_logical_operator_on_mixed_kinds_stays_a_loud_refusal():
    # ``x && p`` is a Number or a Boolean depending on x: no single kind.
    lowered = lower_javascript_source_to_ssa(
        "/** @param {number} x @param {boolean} p */ "
        "function root(x, p) { return x && p; }"
    )
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
    # A label on a block is not a loop's: its exit has no flag.
    "/** @param {number} n */ function root(n) { let s = 0; "
    "blk: { if (n > 1) { break blk; } s = 1; } return s; }",
    # A returned value whose kind is not known before the loop.
    "/** @param {number} n */ function root(n) { "
    "for (let i = 0; i < n; i++) { if (i > 2) return f(i); } return 1; }",
    # A loop with neither a condition nor a break never leaves.
    "/** @param {number} n */ function root(n) { let s = 0; "
    "for (;;) { s += n; } return s; }",
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


@needs_node
def test_module_carries_its_identity_book_through_emission():
    from src.compiler.emission_concordance import ssa_block_cell, value_cell
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    lowered = lower_javascript_source_to_ssa(
        "/** @param {number} n @param {boolean} p */ function root(n, p) "
        "{ let s = 0; for (let i = 0; i < n; i++) "
        "{ if (p) { s += i % 3; } else { s = Math.max(s, i); } } return s; }"
    )
    assert lowered.complete, lowered.shortfall_report()
    module = lowered.module
    book = module.metadata["identity_book"]
    assert not getattr(book, "detached", False)
    function = module.functions["root"]
    values = list(function.args) + [
        instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    ]
    # Every value and block is on the book the interpreter posted to.
    assert all(value_cell(book, function, value) is not None for value in values)
    assert all(ssa_block_cell(book, function, label) for label in function.blocks)
    assert book.pages["name_binding"].row_count() > 0
    assert emit_ssa_to_c(module, "root").complete
    # Emission posted onto that book instead of recording gaps.
    assert not module.metadata.get("emission_gaps")
    assert book.pages["emission_function"].row_count() == 1
    assert concordance_report(module).splitlines()[0].endswith(" 0 finding(s)")


@needs_node
def test_lowering_inside_an_open_compile_posts_to_that_book():
    from src.compiler.identity_concordance import (
        begin_identity_book, end_identity_book,
    )

    book, token = begin_identity_book()
    try:
        lowered = lower_javascript_source_to_ssa(
            "/** @param {number} a */ function root(a) { return a % 2; }"
        )
    finally:
        end_identity_book(token)
    assert lowered.module.metadata["identity_book"] is book


@needs_node
def test_break_and_continue_are_flags_on_the_loop():
    source = (
        "/** @param {number} n */ function root(n) { let s = 0; "
        "for (let i = 0; i < n && i < 40; i++) "
        "{ if (i % 2 === 0) continue; if (s > 20) break; s += i; } return s; }"
    )
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    phis = [
        instruction.attributes["binding_name"]
        for instruction in lowered.module.functions["root"].blocks["loop0_header"].instrs
        if instruction.op == "Phi"
    ]
    # The break flag is carried; the continue flag is per iteration.
    assert sorted(name.split("@")[0] for name in phis) == ["%break", "i", "s"]
    assert check_program(source, "[[0], [3], [9], [100], [NaN], [Infinity]]") == "PASS"


@needs_node
def test_return_inside_a_loop_is_a_carried_flag_and_result():
    source = (
        "/** @param {number} n */ function root(n) { let s = 0; "
        "for (let i = 0; i < n && i < 40; i++) { s += i; if (s > 10) return i; } "
        "return -s; }"
    )
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    phis = sorted(
        instruction.attributes["binding_name"]
        for instruction in lowered.module.functions["root"].blocks["loop0_header"].instrs
        if instruction.op == "Phi"
    )
    assert [name.split("@")[0] for name in phis] == ["%result", "%return", "i", "s"]
    assert check_program(source, "[[0], [3], [9], [100], [NaN], [Infinity]]") == "PASS"


@needs_node
@pytest.mark.parametrize("body", [
    # break/continue to an enclosing loop from a nested one.
    "let s = 0; outer: for (let i = 0; i < 5; i++) { for (let j = 0; j < 5; j++) "
    "{ if (i * j > n) break outer; if (j > i) continue outer; s += j; } s += 100; } return s;",
    # Three levels, each exit to a different loop, beside a return.
    "let s = 0; a: for (let i = 0; i < 4; i++) { b: for (let j = 0; j < 4; j++) "
    "{ for (let k = 0; k < 4; k++) { if (k > j) continue b; if (i + j + k > n) break a; "
    "if (s > 40) return -s; s += 1; } s += 10; } s += 100; } return s;",
])
def test_labeled_exits_set_the_target_loops_flags(body):
    source = f"/** @param {{number}} n */ function root(n) {{ {body} }}"
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    assert check_program(source, "[[0], [2], [5], [9], [NaN], [Infinity]]") == "PASS"


@needs_node
def test_var_that_reads_like_let_is_rewritten_into_one():
    source = (
        "/** @param {number} n */ function root(n) { var s = 0; "
        "for (var i = 0; i < n && i < 40; i++) { var t = i * 2; s += t; } "
        "var n = n + i; return s + n; }"
    )
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    assert check_program(source, "[[0], [3], [9], [100], [NaN], [Infinity]]") == "PASS"


@needs_node
@pytest.mark.parametrize("body", [
    # Read before the declaration runs: the hoisted value is undefined.
    "let s = x; var x = n; return s;",
    # Read in its own initializer.
    "var x = x + n; return x;",
    # Read outside the block that declares it.
    "if (n > 0) { var x = n; } return x;",
    # Declared only in a loop body, read after the loop.
    "for (let i = 0; i < 3; i++) { var x = i; } return x;",
    # No initializer: undefined until assigned.
    "var x; x = n; return x;",
])
def test_var_that_reads_its_hoisted_undefined_stays_a_loud_refusal(body):
    source = f"/** @param {{number}} n */ function root(n) {{ {body} }}"
    lowered = lower_javascript_source_to_ssa(source)
    assert not lowered.complete
    assert any(
        item.code == "unsupported-binding-kind" for item in lowered.shortfalls
    ), lowered.shortfall_report()


@needs_node
@pytest.mark.parametrize("body", [
    "let x = n; if (p) { if (n > 1) return 1; x = 2; } return x;",
    "let x = n; if (p) { x = 1; } else { if (n > 2) return 5; x = 2; } return x * 2;",
    "let x = n; { if (n > 1) return 1; x = 2; } return x;",
])
def test_a_nested_early_return_leaves_the_other_paths_the_rest(body):
    source = (
        f"/** @param {{number}} n @param {{boolean}} p */ "
        f"function root(n, p) {{ {body} }}"
    )
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    assert check_program(
        source, "[[0, true], [3, true], [0, false], [3, false], [NaN, true]]",
    ) == "PASS"


@needs_node
def test_a_nested_early_return_with_no_return_after_it_stays_refused():
    source = (
        "/** @param {number} n @param {boolean} p */ "
        "function root(n, p) { if (p) { if (n > 1) return 1; } }"
    )
    assert not lower_javascript_source_to_ssa(source).complete


@needs_node
@pytest.mark.parametrize("body", [
    "let s = 0; let i = 0; do { s += n; i++; if (s > 9) break; } while (i < 3); return s;",
    "let s = 0; switch (n) { case 1: s = 1; case 2: s += 2; break; default: s = 7; case 3: s += 3; } return s;",
])
def test_do_and_switch_become_the_loops_and_ifs_they_mean(body):
    source = f"/** @param {{number}} n */ function root(n) {{ {body} }}"
    lowered = lower_javascript_source_to_ssa(source)
    assert lowered.complete, lowered.shortfall_report()
    assert check_program(source, "[[0], [1], [2], [3], [5], [NaN], [-0]]") == "PASS"


@needs_node
def test_a_switch_whose_cases_declare_a_let_stays_refused():
    # Every case shares one scope; per-case blocks would split it.
    source = (
        "/** @param {number} n */ function root(n) { switch (n) "
        "{ case 1: let t = 2; return t; default: return 0; } }"
    )
    assert not lower_javascript_source_to_ssa(source).complete
