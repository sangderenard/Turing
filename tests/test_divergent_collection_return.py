"""Return sites that return different lists.

``return [n]`` on one edge and ``return []`` on another are two arenas;
the caller has one result.  A list born in the function dies at its
return, so each edge copies its list into the one result list (lanes plan
A6; corpus c541-c544).  A parameter's list returned on one edge is the
caller's own object: copying it would split one object in two, so that is
still refused by name until the caller-side alias check exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def test_a_returned_parameter_list_beside_a_local_list_is_refused_by_name():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def pick(xs: list[int], n: int) -> list[int]:\n"
        "    out = [n, n + 1]\n"
        "    if n > 2:\n"
        "        return out\n"
        "    return xs\n"
        "\n\n"
        "def root(n: int) -> int:\n"
        "    base = [7, 8, 9]\n"
        "    t = pick(base, n)\n"
        "    return len(t) * 100 + t[0] * 10 + t[-1]\n"
    )
    with pytest.raises(ValueError, match="only lists born in the function"):
        lower_ast_source_to_ssa(
            source, "root", name="x", extraction_contract=CONTRACT,
            progress=lambda _: None,
        )
