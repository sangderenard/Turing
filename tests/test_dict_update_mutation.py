"""``d.update(...)`` is a recorded table mutation or a named refusal.

``d.update(<dict display>)`` stores the display's rows (corpus c214).  An
update from a mapping that changed after its birth, or with keywords, has
no fixed rows and is refused by name rather than dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


@pytest.mark.parametrize("source", [
    "def root(n: int) -> int:\n"
    "    d = {1: 10, 2: 20}\n"
    "    e = {2: n}\n"
    "    e[2] = 9\n"
    "    d.update(e)\n"
    "    return d[2] * 10 + n\n",
    "def root(n: int) -> int:\n"
    "    d = {1: 10, 2: 20}\n"
    "    d.update(a=n)\n"
    "    return d[1] + n\n",
])
def test_update_without_fixed_rows_is_refused(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with pytest.raises(Exception, match="update"):
        lower_ast_source_to_ssa(
            source, "root", name="x", extraction_contract=CONTRACT,
            progress=lambda _: None,
        )
