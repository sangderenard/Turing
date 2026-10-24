"""A local ``dict[int, Any]`` declares an opaque value column.

``SSAReferenceEvaluator.run`` builds ``values: dict[int, Any] = {}`` and
stores ``arguments[key]`` from a ``Mapping[int, Any]`` parameter.  The local
annotation used to declare the value column as the int64 handle of a record
named ``'Any'``, so the store of the opaque value refused as a row dtype
disagreement.
"""

from __future__ import annotations

from src.compiler.project_compilation_product import compile_project_call


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
