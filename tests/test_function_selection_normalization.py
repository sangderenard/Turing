"""A call through a choice of module functions is a choice of calls."""

from __future__ import annotations

import ast

import pytest

FUNCTIONS = (
    "def inc(x):\n"
    "    return x + 1\n"
    "def neg(x):\n"
    "    return -x\n"
)


def _normalized(body: str) -> tuple[str, tuple]:
    from src.compiler.ast_control_normalization import (
        _normalize_function_selection,
    )

    tree = ast.parse(FUNCTIONS + body)
    receipts = _normalize_function_selection(tree)
    return ast.unparse(tree), receipts


def _run(text: str, **inputs):
    namespace: dict = {}
    exec(compile(text, "<normalized>", "exec"), namespace)
    return namespace["root"](**inputs)


@pytest.mark.parametrize("body", [
    "def root(a, i):\n"
    "    f = inc if i == 0 else neg\n"
    "    return f(a) * 10 + a\n",
    "def root(a, i):\n"
    "    if i == 0:\n"
    "        f = inc\n"
    "    else:\n"
    "        f = neg\n"
    "    return f(a) * 10 + a\n",
    "def root(a, i):\n"
    "    f = inc\n"
    "    if i > 0:\n"
    "        f = neg\n"
    "    return f(a) * 10 + a\n",
    "def root(a, i):\n"
    "    return (inc if i == 0 else neg)(a) * 10 + a\n",
])
def test_selected_callee_becomes_a_choice_of_direct_calls(body):
    text, receipts = _normalized(body)
    assert receipts
    assert "f(" not in text.replace("def ", "")
    for a, i in ((3, 0), (7, 1), (-2, 0)):
        assert _run(text, a=a, i=i) == _run(FUNCTIONS + body, a=a, i=i)


@pytest.mark.parametrize("body", [
    "def apply(g, x):\n"
    "    return g(x)\n"
    "def root(a, i):\n"
    "    f = inc if i == 0 else neg\n"
    "    return apply(f, a)\n",
    "def root(a, i):\n"
    "    f = inc if i == 0 else neg\n"
    "    g = f\n"
    "    return g(a)\n",
])
def test_a_function_choice_used_as_a_value_is_refused(body):
    with pytest.raises(ValueError, match="function_selection_value"):
        _normalized(body)


def test_a_rebound_function_name_is_not_a_known_function():
    text, receipts = _normalized(
        "inc = neg\n"
        "def root(a, i):\n"
        "    return a\n"
    )
    assert receipts == ()
