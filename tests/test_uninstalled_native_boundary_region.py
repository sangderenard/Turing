"""A region of ``use_native`` calls no boundary installer retains computes.

``math.floor`` is classified ``use_native`` with no ``result_dtype``, so
``_install_external_reference_calls`` leaves it unplaced; the region holding
only that call was still retired as boundary-owned, and its value vanished:
``return math.floor(x), int(x)`` returned one value (corpus c922) and
``return math.floor(x)`` was refused ("returns none").  An imported
``floor`` is ``math.floor`` by identity and returns an int (corpus c923).
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"


def _returned_dtypes(body: str) -> list[str]:
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "import math\nfrom math import ceil, floor, trunc\n\n\n"
        f"def root(x: float):\n    {body}\n"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="uninstalled_native_boundary",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    (ret,) = [
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ]
    return [str(value.dtype) for value in ret.args]


@pytest.mark.parametrize("body, dtypes", [
    ("return math.floor(x), int(x)", ["int64", "int64"]),
    ("return int(x), math.ceil(x)", ["int64", "int64"]),
    ("return math.floor(x)", ["int64"]),
    ("return math.floor(x), math.ceil(x)", ["int64", "int64"]),
    ("return math.sqrt(x), int(x)", ["float64", "int64"]),
    ("return floor(x), ceil(x), trunc(x)", ["int64", "int64", "int64"]),
])
def test_every_returned_value_survives_with_its_python_class(body, dtypes):
    assert _returned_dtypes(body) == dtypes
