"""A function whose return sites return different lists is refused.

``return [n]`` on one edge and ``return []`` on another are two arenas; the
caller has one result.  Until the variadic sequence return is lowered (lanes
plan A6) the planner refuses it by name.  It used to bind only the last
return site and read the empty list on every path.  The single-display
return is corpus c215.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def test_two_returned_lists_are_refused_by_name():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def pick(n: int) -> list[int]:\n"
        "    if n > 2:\n"
        "        return [n]\n"
        "    return []\n"
        "\n\n"
        "def root(n: int) -> int:\n"
        "    t = pick(n)\n"
        "    return len(t) * 10 + (t[0] if t else 0)\n"
    )
    with pytest.raises(ValueError, match="returns different collections"):
        lower_ast_source_to_ssa(
            source, "root", name="x", extraction_contract=CONTRACT,
            progress=lambda _: None,
        )
