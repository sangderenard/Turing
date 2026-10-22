"""Chained comparisons: which middle operands may be spelled twice.

A middle operand of ``a < m < b`` is evaluated once by Python.  The
normalizer spells the chain as ``a < m and m < b``, so it may duplicate
``m`` only when a second evaluation is the first one's value with no
effect: a pure read, or a builtin conversion or measure of one.  Every
other middle (a user call, a method call, a shadowed builtin) is refused
by name.  Native value checks live in corpus c155, c211 and c212.
"""

from __future__ import annotations

import ast

import pytest

from src.compiler.ast_control_normalization import (
    _normalize_chained_comparisons,
)


def _normalized(source: str) -> tuple[str, tuple]:
    tree = ast.parse(source)
    receipts = _normalize_chained_comparisons(tree)
    return ast.unparse(tree), receipts


def test_guarded_conversion_middle_stays_inside_the_guard():
    text, receipts = _normalized(
        "def f(d):\n"
        "    return d.isdigit() and 0 <= int(d) <= 9\n"
    )
    # Both occurrences sit after the guard: nothing is hoisted.
    assert "d.isdigit() and (0 <= int(d) and int(d) <= 9)" in text
    assert receipts[0]["reevaluated_conversions"] == ["int(d)"]


@pytest.mark.parametrize("middle", [
    "float(v)", "len(xs)", "abs(int(e))", "ord(c[0])", "int(p.q)",
])
def test_builtin_conversion_middles_are_spelled(middle):
    text, _ = _normalized(f"r = 0 < {middle} < 9\n")
    assert text == f"r = 0 < {middle} and {middle} < 9"


@pytest.mark.parametrize("source", [
    "def g(k):\n    return k\nr = 0 < g(1) < 9\n",
    "r = 0 < v.dim() < 9\n",
    "r = 0 < int(g(1)) < 9\n",
    "r = 0 < int(v, 2) < 9\n",
    "r = 0 < int(*v) < 9\n",
    "def int(v):\n    return v\nr = 0 < int(1) < 9\n",
    "from m import f as len\nr = 0 < len(v) < 9\n",
    "def f(abs):\n    return 0 < abs(1) < 9\n",
])
def test_other_call_middles_are_refused(source):
    with pytest.raises(ValueError, match="middle operand with effects"):
        _normalized(source)


def test_call_as_outer_operand_is_evaluated_once():
    text, receipts = _normalized("r = g(1) < x < h(2)\n")
    assert text == "r = g(1) < x and x < h(2)"
    assert receipts[0]["reevaluated_conversions"] == []
