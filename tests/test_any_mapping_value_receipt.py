"""``Mapping[K, Any]`` declares an opaque-handle value, never a record.

``Any`` is the repository's fixed-width opaque handle as a parameter, a record
field and a collection member.  A mapping's value fell through to the record
branch instead and named a record ``'Any'`` that no declaration defines, so
``SSAReferenceEvaluator.run`` (``arguments: Mapping[int, Any]``) refused with
"top-level keyed value_record requires its existing record-row binding".
"""

from __future__ import annotations

import ast

import pytest

from src.compiler.fortran_c_shell import _authored_annotation_field_receipt


@pytest.mark.parametrize("spelling", [
    "Mapping[int, Any]", "dict[str, Any]", "dict[int, object]",
])
def test_arbitrary_object_mapping_value_is_an_opaque_handle(spelling):
    receipt = _authored_annotation_field_receipt(
        ast.parse(spelling, mode="eval").body,
    )
    assert receipt["storage"] == "keyed"
    assert receipt["dtype"] == "opaque_ref"
    assert "value_record" not in receipt


def test_declared_record_value_still_names_its_record():
    receipt = _authored_annotation_field_receipt(
        ast.parse("dict[int, Point]", mode="eval").body,
    )
    assert receipt["value_record"] == "Point"
