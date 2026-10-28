"""Iterating a table with a keyed delete skips the deleted rows.

``del d[k]`` clears the row's live flag; the row stays resident.  A loop
over the table ran over the physical rows and visited the deleted key
silently (lanes plan C3), so it was refused by name.  It is lowered now:
the loop reads each row's live flag and takes the latch for a cleared one
(corpus c716, c717 check the values).
"""

from __future__ import annotations

from pathlib import Path

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def _lower(source: str):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    return lower_ast_source_to_ssa(
        source, "root", name="x", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )


def _block_names(module) -> set[str]:
    return {
        getattr(block, "name", block)
        for function in module.functions.values()
        for block in function.blocks
    }


def test_iteration_over_a_table_with_a_keyed_delete_reads_the_live_flag():
    body = (
        "    d = {1: 10, 2: 20, 3: 30}\n    del d[2]\n    t = 0\n"
        "    for k in d:\n        t += k\n    return t * 10 + n\n"
    )
    module, _, _ = _lower("def root(n: int) -> int:\n" + body)
    assert any(name.startswith("table_row_live") for name in _block_names(module))
