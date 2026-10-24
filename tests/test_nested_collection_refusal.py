"""A list whose rows are lists is refused by name.

A resident row is a scalar or a tuple's leaves.  A list display with a list
member had no row birth; the inner display's member birth and its own
``clear`` met on one control block row and the program was refused with an
anonymous uncaused revision (lanes plan C6).  Reads of a folded nested
display (``g[1][1]``) do not reach the birth and still lower.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def _lower(source: str) -> None:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    lower_ast_source_to_ssa(
        source, "root", name="x", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )


@pytest.mark.parametrize("body", [
    "    g = [[0, 0], [0, 0]]\n    g[1][0] = n\n    return g[1][0]\n",
    "    g = [[0, 0], [0, 0]]\n    g[1] = [n, 1]\n    return g[1][0]\n",
    "    t = 0\n    for row in [[1, 2], [3, n]]:\n"
    "        t += row[0] * row[1]\n    return t\n",
])
def test_a_list_of_lists_is_refused_by_name(body):
    with pytest.raises(ValueError, match="rows are collections"):
        _lower("def root(n: int) -> int:\n" + body)
