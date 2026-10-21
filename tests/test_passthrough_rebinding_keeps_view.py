"""A pass-through rebinding keeps the declared view of the storage.

Program: ``valid.reshape((-1, 1, 1))`` planned as a call whose output is a view
of the callee formal ``valid`` and consumed by a later region.  The aggregate
ABI legalization folds that output into the call's own actual.  The rebinding
must carry the callee descriptor's declared view (``ssa_storage_view`` with
``view_shape (8, 1, 1)``), the ``aggregate_passthrough_rebinding`` row must
state that view shape, and no ``proven_shape`` row of the lowering may be
``conflicting`` (a bare rebinding was restamped to the storage's ``(8,)``
against the consumer formal's ``(8, 1, 1)``).
"""
from __future__ import annotations

import numpy as np

from src.compiler.concordance_declarations import (
    AGGREGATE_PASSTHROUGH_REBINDING,
)
from src.compiler.identity_concordance import identity_book

from tests.test_aggregate_call_identity import _lower

SOURCE = """
def inner(inputs, valid, count):
    keep = AbstractTensor.where(valid.reshape((-1, 1, 1)), inputs, inputs * 0.0)
    total = inputs * 0.0
    for step in range(count):
        total = total + keep
    return total, valid * 0.0 + 1.0


def tick(inputs, valid, count):
    total, following = inner(inputs, valid, count)
    return total, following
"""

FEEDS = {"inputs": np.zeros((8, 4, 3)), "valid": np.zeros((8,)), "count": 1}


def test_passthrough_rebinding_keeps_declared_view():
    module, _outputs, _exports = _lower(SOURCE, "passthrough_view", FEEDS)
    book = identity_book(module)

    page = book.page(AGGREGATE_PASSTHROUGH_REBINDING)
    views = [
        (row, page.latest(row)) for row in page.rows()
        if len(page.latest(row)) == 4
    ]
    assert views, "no rebinding row states a view shape"
    for row, fact in views:
        assert tuple(fact[3]) == (8, 1, 1), (row, fact)

    # The rebinding occurrence is later replaced by the descriptor-derived
    # view occurrence (it keeps the declaration, not the passthrough marker),
    # so the declaration is read from the final args and from the book rows:
    # the rebinding row above states the view shape, and the call feed's
    # producer endpoint is the view's own structural_value row.
    carried = [
        argument
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        for argument in instruction.args
        if (argument.accounting or {}).get("ssa_storage_view")
    ]
    assert carried, "no final occurrence carries ssa_storage_view"
    for argument in carried:
        view = argument.accounting["ssa_storage_view"]
        assert tuple(view["view_shape"]) == (8, 1, 1)

    structural = book.page("structural_value")
    view_rows = [
        (row, structural.latest(row)) for row in structural.rows()
        if "ssa_storage_view" in repr(row[0])
    ]
    assert view_rows, "no structural_value row for the declared view"
    for row, value in view_rows:
        assert tuple(value.shape) == (8, 1, 1), (row, value)

    proven = book.page("proven_shape")
    conflicting = [
        (row, stored)
        for row in proven.rows()
        for _column, stored in proven.history(row)
        if isinstance(stored, tuple) and stored and stored[0] == "conflicting"
    ]
    assert not conflicting, conflicting
