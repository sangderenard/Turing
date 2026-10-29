"""Every position of a returned tuple is its own Ret slot, repeats included.

Program: ``return (x, (x, 1.0))`` (corpus c924).  Structural output recovery
assembled the Ret by value id: the inner tuple's ``x`` was "already
published" by the outer ``x``, and the order was a sort by first rank, so C,
LLVM and Fortran returned ``(x, 1.0)`` for Python's leaves ``x, x, 1.0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _ret_ids(body: str) -> list[int]:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        f"def root(x: float):\n    {body}\n", "root",
        name="repeated_return_leaf", extraction_contract=str(CONTRACT),
        progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    (ret,) = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]
    return [int(value.id) for value in ret.args]


@pytest.mark.parametrize("body, pattern", [
    ("return (x, (x, 1.0))", "aab"),
    ("return ((x, 1.0), x)", "aba"),
    ("return (x, (2.0, x))", "aba"),
    ("return x, x", "aa"),
    ("return (x, (x + 1.0, 2.0))", "abc"),
])
def test_each_returned_position_has_its_own_slot(body, pattern):
    ids = _ret_ids(body)
    # The Ret's value identities follow the source's leaf pattern: equal
    # letters are one value at several positions, never collapsed.
    letters = {}
    assert "".join(
        letters.setdefault(value_id, "abc"[len(letters)]) for value_id in ids
    ) == pattern
