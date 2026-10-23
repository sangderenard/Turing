"""Iterating a table with a keyed delete is refused by name.

``del d[k]`` clears the row's live flag; the row stays resident.  A loop
over the table ran over the physical rows and visited the deleted key
silently (lanes plan C3).
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


def test_iteration_over_a_table_with_a_keyed_delete_is_refused_by_name():
    body = (
        "    d = {1: 10, 2: 20, 3: 30}\n    del d[2]\n    t = 0\n"
        "    for k in d:\n        t += k\n    return t * 10 + n\n"
    )
    with pytest.raises(ValueError, match="iteration over a table with deletions"):
        _lower("def root(n: int) -> int:\n" + body)
