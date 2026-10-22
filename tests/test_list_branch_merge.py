"""A list bound on two branches lowers only from unshared equal displays.

``if c: t = [n, 1] else: t = [5, 6]`` is born at the merge from its member
Phis (corpus c271).  That copy is the list only while each arm is its own
unchanged display, and only arms of equal length publish members.  Every
other shape used to read a never-born arena or a stale copy silently; it is
now refused by name.
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


@pytest.mark.parametrize("source, message", [
    (
        "def root(n: int) -> int:\n"
        "    if n > 2:\n"
        "        t = [n]\n"
        "        t.append(1)\n"
        "    else:\n"
        "        t = [5]\n"
        "    return len(t) * 10 + t[0]\n",
        "changes or is shared",
    ),
    (
        "def root(n: int) -> int:\n"
        "    a = [1, 2]\n"
        "    if n > 2:\n"
        "        t = a\n"
        "    else:\n"
        "        t = [3, 4]\n"
        "    t.append(9)\n"
        "    return len(a) * 10 + len(t)\n",
        "changes or is shared",
    ),
    (
        "def pick(n: int) -> list[int]:\n"
        "    if n > 2:\n"
        "        r = [n]\n"
        "    else:\n"
        "        r = []\n"
        "    return r\n"
        "\n\n"
        "def root(n: int) -> int:\n"
        "    t = pick(n)\n"
        "    return len(t) * 10 + (t[0] if t else 0)\n",
        "different shapes on two branches",
    ),
])
def test_unlowerable_branch_merged_list_is_refused_by_name(source, message):
    with pytest.raises(ValueError, match=message):
        _lower(source)
