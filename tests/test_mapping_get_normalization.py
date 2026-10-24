"""``d.get`` on a proven mapping is the keyed selection Python defines."""

from __future__ import annotations

import ast

from src.compiler.ast_control_normalization import _normalize_mapping_get


def _rewrite(source: str) -> tuple[str, int]:
    tree = ast.parse(source)
    receipts = _normalize_mapping_get(tree)
    return ast.unparse(tree), len(receipts)


def test_get_on_a_dict_display_becomes_a_membership_selection():
    text, count = _rewrite(
        "def root(n):\n"
        "    d = {1: n}\n"
        "    m = n * 3\n"
        "    return d.get(n), d.get(n, m), d.get(n, 9)\n"
    )
    assert count == 2
    assert "d[n] if n in d else None" in text
    assert "d[n] if n in d else m" in text
    # A numeric literal default keeps the exact lookup-or-default helper.
    assert "d.get(n, 9)" in text


def test_get_on_an_annotated_mapping_parameter_is_rewritten():
    text, count = _rewrite(
        "def f(table: dict[int, int], k: int):\n"
        "    return table.get(k)\n"
    )
    assert count == 1
    assert "table[k] if k in table else None" in text


def test_get_on_an_unproven_receiver_is_kept():
    text, count = _rewrite(
        "def f(store, k, items):\n"
        "    cache = {}\n"
        "    for cache in items:\n"
        "        pass\n"
        "    other = make()\n"
        "    return store.get(k), cache.get(k), other.get(k), cache.get(k + f(k))\n"
    )
    assert count == 0
    assert text.count(".get(") == 4
