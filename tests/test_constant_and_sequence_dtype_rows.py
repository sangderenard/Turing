"""A resident SSAValue's dtype is a view of its ``ssa_value`` row.

Programs: ``return None`` (a NoneValue materialized "none" over a row adopted
with dtype None) and ``sorted(generator)`` (sort scratch / result arenas whose
rows were adopted 'unknown' while the resident took the concorded dtype).
"""
from pathlib import Path

import pytest

from src.compiler.concordance_declarations import SSA_VALUE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_record_return_state import function_scope_of

CONTRACT = Path("extraction_contracts/program_extraction.yaml")

RETURN_NONE = """
def f(x: int) -> int:
    return None
"""

SORTED_GENERATOR = """
def f(xs: list[int]) -> list[int]:
    return sorted(x + 1 for x in xs)
"""


def _lower(source):
    module, _, _ = lower_ast_source_to_ssa(
        source, "f", extraction_contract=ExtractionContract(CONTRACT),
        progress=lambda message: None, progress_bars=False,
    )
    return module


def _row_dtype(module, function, value):
    book = identity_book(module)
    row = (function_scope_of(function), int(value.id))
    ref = book.latest_ref(SSA_VALUE, row)
    assert ref is not None, (function.name, value.id)
    return book.pages[SSA_VALUE.name].cells.get((row, ref.column)).dtype


def _residents(function):
    seen = {}
    for value in function.args:
        seen[int(value.id)] = value
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                seen[int(instruction.res.id)] = instruction.res
    return seen.values()


def test_none_value_row_matches_materialized_resident():
    module = _lower(RETURN_NONE)
    checked = 0
    for function in module.functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) != "NoneValue":
                    continue
                checked += 1
                assert instruction.res.dtype == "none"
                assert _row_dtype(module, function, instruction.res) == "none"
    assert checked


def test_sorted_generator_arena_rows_match_residents():
    module = _lower(SORTED_GENERATOR)
    arenas = 0
    for function in module.functions.values():
        for value in _residents(function):
            if not (value.accounting or {}).get("sequence_arena"):
                continue
            arenas += 1
            assert _row_dtype(module, function, value) == value.dtype, (
                function.name, value.id,
            )
    assert arenas
