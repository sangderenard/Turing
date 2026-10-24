"""A local ``dict[int, Any]`` declares an opaque value column.

``SSAReferenceEvaluator.run`` builds ``values: dict[int, Any] = {}`` and
stores ``arguments[key]`` from a ``Mapping[int, Any]`` parameter.  The local
annotation used to declare the value column as the int64 handle of a record
named ``'Any'``, so the store of the opaque value refused as a row dtype
disagreement.
"""

from __future__ import annotations

import pytest

from src.compiler.project_compilation_product import compile_project_call
from src.compiler.ssa_fortran_backend import FortranEmissionError


SOURCE = '''
from typing import Any, Mapping


def run(ids: list[int], arguments: Mapping[int, Any]) -> int:
    values: dict[int, Any] = {}
    for value in ids:
        key = int(value)
        if key in arguments:
            values[key] = arguments[key]
    return len(values)
'''


def test_any_valued_local_mapping_stores_the_opaque_value(tmp_path):
    source = tmp_path / "any_rows.py"
    source.write_text(SOURCE, encoding="utf-8")
    receipt = compile_project_call(
        source, "run", tmp_path / "unit", progress_bars=False,
    )
    assert receipt["repository_ssa_complete"]


GROWN_FIELD = '''
from typing import Any, Mapping


class Rec:
    def __init__(self) -> None:
        self._root_frame: dict[int, Any] | None = None

    def run(self, ids: list[int], arguments: Mapping[int, Any]) -> int:
        values: dict[int, Any] = {}
        self._root_frame = values
        for key in ids:
            if key in arguments:
                values[int(key)] = arguments[key]
        return len(values)
'''


def test_keyed_field_assigned_a_grown_mapping_is_refused_by_field(tmp_path):
    # The keyed field's view is caller-supplied and exactly full (length and
    # capacity are one part); the frame's grown arena used to die in the
    # heap descriptor invariant with no name.
    source = tmp_path / "grown_field.py"
    source.write_text(GROWN_FIELD, encoding="utf-8")
    with pytest.raises(FortranEmissionError, match=(
        r"keyed field Rec\._root_frame is assigned a mapping this frame grows"
    )):
        compile_project_call(
            source, "Rec.run", tmp_path / "unit", progress_bars=False,
        )
