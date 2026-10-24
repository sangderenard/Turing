"""``set(it)``/``frozenset(it)`` are the set of the members ``it`` yields."""

from __future__ import annotations

import ast


def test_set_and_frozenset_of_one_iterable_become_set_comprehensions():
    from src.compiler.ast_control_normalization import (
        _normalize_set_constructor,
    )

    tree = ast.parse(
        "def root(n):\n"
        "    xs = [n, 1]\n"
        "    return set(xs), frozenset(int(v) for v in xs if v), set()\n"
    )
    receipts = _normalize_set_constructor(tree)
    text = ast.unparse(tree)
    assert len(receipts) == 2
    assert "{_turing_set_member for _turing_set_member in xs}" in text
    assert "{int(v) for v in xs if v}" in text
    # The empty constructor has no members to spell.
    assert "set()" in text


def test_a_module_binding_set_keeps_its_calls():
    from src.compiler.ast_control_normalization import (
        _normalize_set_constructor,
    )

    tree = ast.parse(
        "def set(xs):\n"
        "    return xs\n"
        "def root(n):\n"
        "    return set([n]), frozenset([n])\n"
    )
    receipts = _normalize_set_constructor(tree)
    text = ast.unparse(tree)
    assert len(receipts) == 1
    assert "set([n])" in text
    assert "frozenset" not in text
