"""A parameter's descriptor members keep the storage kind their row carries.

``for k in d`` over ``d: dict[int, int]`` and ``for a, b in ps`` over
``ps: list[tuple[int, int]]`` stopped at the physical view with ``span``
versus ``sequence`` / ``reference``: the sequence descriptor restated
``span`` over a member the ProgramABI binding had already posted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


@pytest.mark.parametrize("source", [
    (
        "def root(d: dict[int, int]) -> int:\n"
        "    t = 0\n"
        "    for k in d:\n"
        "        t += k\n"
        "    return t\n"
    ),
    (
        "def root(ps: list[tuple[int, int]]) -> int:\n"
        "    t = 0\n"
        "    for a, b in ps:\n"
        "        t += a * b\n"
        "    return t\n"
    ),
])
def test_parameter_member_storage_is_not_restated(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    lower_ast_source_to_ssa(
        source, "root", name="x", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )


def test_dict_items_loop_does_not_duplicate_the_column_formal():
    """``for k, v in d.items()`` reads the descriptor's own column cell.

    The column was requested by id and appended as a second formal, so the
    callee's signature repeated one id and its callers' input-binding receipt
    was refused ("receipt=6, operands=6, unique_formals=5").
    """

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    lower_ast_source_to_ssa(
        "def total(d: dict[int, int]) -> int:\n"
        "    t = 0\n"
        "    for k, v in d.items():\n"
        "        t += k * v\n"
        "    return t\n"
        "\n"
        "def root(n: int) -> int:\n"
        "    return total({1: n, 2: 3})\n",
        "root", name="x", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )
