"""Nested loops over a 2-D span lower with zero concordance findings.

Two sources: ``for`` inside ``for`` accumulating into a scalar and into a
1-D span, and a ``while`` inside a ``for``.  Both reached
``shape_transformation_concordance REVISE without a changed source``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_aggregate_call_identity import _lower

FOR_FOR = """
def tick(a, out):
    total = 0.0
    for i in range(8):
        for j in range(4):
            out[i] = out[i] + a[i, j]
            total = total + a[i, j]
    return total, out
"""

FOR_WHILE = """
def tick(a, out):
    for i in range(8):
        j = 0
        while j < 4:
            out[i] = out[i] + a[i, j]
            j = j + 1
    return out
"""

FEEDS = {"a": np.zeros((8, 4)), "out": np.zeros((8,))}


@pytest.mark.parametrize("name,source", [("for_for", FOR_FOR), ("for_while", FOR_WHILE)])
def test_nested_loops_lower_without_concordance_findings(name, source):
    module, _outputs, _exports = _lower(source, "nested_" + name, FEEDS)
    assert module.functions


CARRIED_SCALAR_WHILE = """
def tick(a, out):
    for i in range(8):
        total = 0.0
        j = 0
        while j < 4:
            total = total + a[i, j]
            j = j + 1
    return total
"""


def test_while_header_phi_row_dtype_is_the_posted_dtype_of_its_literal():
    # Program: ``j = 0`` carried through ``while j < 4`` inside a ``for``.
    # Wrong record: the while_header Phi was minted with the literal's
    # authored spelling (``int``) while the literal's posted ssa_value row is
    # the declared ``int64`` (constant materialization revises it), so the
    # shape edge keyed on the Phi id received ``int64 -> int`` beside the
    # region-call occurrences' ``int64 -> int64`` and REVISE refused it.
    # Rule: a carried Phi is minted from its initial's POSTED row (dtype and
    # physical fields), derived from that cell; no reader revises the
    # occurrence afterwards.
    from src.compiler.concordance_declarations import SSA_VALUE
    from src.compiler.identity_concordance import identity_book
    from src.compiler.ssa_record_return_state import function_scope_of

    module, _outputs, _exports = _lower(
        CARRIED_SCALAR_WHILE, "carried_while", FEEDS,
    )
    book = identity_book(module)

    def row_dtype(function, value):
        row = (function_scope_of(function), int(value.id))
        ref = book.latest_ref(SSA_VALUE, row)
        assert ref is not None, (function.name, value.id)
        return book.pages[SSA_VALUE.name].cells.get((row, ref.column)).dtype

    checked = 0
    for function in module.functions.values():
        for label, block in function.blocks.items():
            if not str(label).startswith("while_header"):
                continue
            for instruction in block.instrs:
                if str(instruction.op) != "Phi" or instruction.res is None:
                    continue
                if instruction.res.dtype in {"bool", "ptr"}:
                    continue
                seed = instruction.args[0]
                checked += 1
                assert row_dtype(function, instruction.res) == row_dtype(
                    function, seed
                ), (function.name, label, instruction.res.id, seed.id)
                assert instruction.res.dtype == row_dtype(
                    function, instruction.res
                )
    assert checked
