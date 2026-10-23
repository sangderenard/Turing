"""A list replication lowers only as the fill of one member.

``[0] * k`` and ``[n] * k`` are born by their ``fill`` initialization.  A
wider row used to lower as a numerical ``Mul`` of the row's handle, so
``[1, n] * 3`` read past the row silently (lanes plan C9); it is now refused
by name.
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
    "    xs = [1, n] * 3\n    return len(xs) * 100 + xs[3] * 10 + xs[4]\n",
    "    xs = 2 * [n, 4, n + 1]\n    return xs[5]\n",
    "    xs = [1, n] * (n % 3 + 1)\n    return len(xs)\n",
    "    xs = (1, n) * 2\n    return len(xs) * 10 + xs[3]\n",
])
def test_replication_of_a_wider_row_is_refused_by_name(body):
    with pytest.raises(ValueError, match="replicates a row other than one member"):
        _lower("def root(n: int) -> int:\n" + body)
