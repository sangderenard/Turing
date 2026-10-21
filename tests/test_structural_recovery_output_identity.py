"""Structural recovery names an output's ROOT storage, not a store version.

Program shape: a function that writes into a parameter inside loops (flat or
nested) and returns it::

    def tick(a, out):
        for i in range(8):
            total = 0.0
            j = 0
            while j < 4:
                total = total + a[i, j]
                j = j + 1
            out[i] = total
        return out

Wrong record: the identity table's last binding of ``out`` is an IndexedStore
version minted by the loop composer (it aliases the root through
``value_aliases``), while the reducer's SEMANTIC_OUTPUT row, the control
builder's revision, the FUNCTION_OUTPUT row and the emitted Ret all name the
root (the formal).  Structural recovery compared the version with the row and
refused.  Rule: a store-chain version and its root are one storage; recovery
resolves the version through the shared alias authority and the declared
output stays the root.  A genuine disagreement still refuses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import CorrelationTable  # noqa: E402
from src.compiler.ssa_reachability import function_output_rows  # noqa: E402
from src.compiler.identity_concordance import identity_book  # noqa: E402

FLAT = '''
def tick(a, out):
    for i in range(8):
        out[i] = a[i]
    return out
'''

NESTED = '''
def tick(a, out):
    for i in range(8):
        total = 0.0
        j = 0
        while j < 4:
            total = total + a[i, j]
            j = j + 1
        out[i] = total
    return out
'''


@pytest.mark.parametrize("source", [FLAT, NESTED], ids=["flat", "nested"])
def test_store_version_output_resolves_to_the_declared_root(source):
    contract = ExtractionContract(
        ROOT / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "tick", extraction_contract=contract,
        progress=lambda message: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    assert CorrelationTable.build(module).findings(module) == []
    rows = function_output_rows(function, book=identity_book(module))
    assert [name for name, _value in rows] == ["out"]
    returned = {
        int(argument.id)
        for block in function.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for argument in instruction.args
    }
    assert {int(value) for _name, value in rows} <= returned
    assert function.metadata["semantic_output_ids"] == tuple(
        int(value) for _name, value in rows
    )
