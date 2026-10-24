"""A list whose rows mix int and float is refused by name, never truncated.

Program: ``ys = [n]; ys.append(n * 0.5); ys[1]`` and
``ys = [1 * n, 2.5 * n]; sum(ys)`` (lanes plan C13).  Wrong record: the
arena took its first row's int64 type and the append helper's call input
adapter cast the float row to int64 (``sum`` read 3.0 for 3.5; clean audit).
Python never converts a float to an int implicitly, so the call input
conversion refuses a float actual bound to an integer formal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

PROGRAMS = {
    "append_float_to_int_list": (
        "def root(n: int) -> float:\n"
        "    ys = [n]\n"
        "    ys.append(n * 0.5)\n"
        "    return ys[1]\n"
    ),
    "display_mixes_int_and_float": (
        "def root(n: int) -> float:\n"
        "    ys = [1 * n, 2.5 * n]\n"
        "    return sum(ys)\n"
    ),
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_mixed_row_list_is_refused_not_truncated(name: str) -> None:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with pytest.raises(Exception, match="call input conversion would truncate"):
        lower_ast_source_to_ssa(
            PROGRAMS[name], "root", name=f"mixed_{name}",
            extraction_contract=str(CONTRACT), progress=lambda _: None,
        )


def test_float_stored_into_int_list_row_is_refused() -> None:
    """``ys = [n, 2]; ys[0] = 0.5`` (Lane D): the float store into the
    int64 arena was narrowed (2 for 2.5); now refused by name."""
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with pytest.raises(Exception, match="float-into-integer-element"):
        lower_ast_source_to_ssa(
            "def root(n: int) -> float:\n"
            "    ys = [n, 2]\n"
            "    ys[0] = 0.5\n"
            "    return ys[0] + ys[1]\n",
            "root", name="mixed_store_float_into_int_row",
            extraction_contract=str(CONTRACT), progress=lambda _: None,
        )
