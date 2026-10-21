"""A planned region shares its parent's scope string; it is not record-covered.

Program: a function with an optional tuple return plus a region cut from it.
Wrong record: the parent's AGGREGATE_RETURN_LAYOUT row (scope, 16), reached
by the region through the shared scope.  Rule: a layout row is the callee's
only if the callee's own Ret returns that output (or its expansion).
"""

import pytest

from src.compiler.concordance_declarations import (
    OPTIONAL_VALUE_STRUCTURE, STRUCTURAL_CONCORDANCE,
)
from src.compiler.identity_concordance import (
    Derived, Mode, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_optional_values import _record_covered_return_receipts
from src.compiler.ssa_record_return_state import function_scope_of, identity_cells
from src.compiler.structural_concordance import publish_aggregate_return_layout
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


@pytest.fixture(autouse=True)
def _fresh_compile_book():
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)


def _functions(parent_ret):
    p0, q0 = SSAValue(17, "int64"), SSAValue(100, "bool")
    p1, q1 = SSAValue(18, "int64"), SSAValue(101, "bool")
    one = SSAValue(1, "int64")
    parent = Function("f", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], one, attributes={"value": 1}),
        Instr("Const", [], p0, attributes={"value": 1}),
        Instr("Const", [], p1, attributes={"value": 2}),
        Instr("Const", [], q0, attributes={"value": 1}),
        Instr("Const", [], q1, attributes={"value": 1}),
        Instr("Ret", [SSAValue(int(v), "int64") for v in parent_ret], None),
    ])}, metadata={"tensor_shape_concordance_scope": "f@control:0",
                   "semantic_output_ids": (16,)})
    inner = SSAValue(4, "int64")
    region = Function("f__planned_region_0", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], inner, attributes={"value": 3}),
        Instr("Ret", [inner], None),
    ])}, metadata={"tensor_shape_concordance_scope": "f@control:0"})
    return parent, region


@pytest.mark.parametrize("parent_ret", [(16,), (17, 100, 18, 101)])
def test_region_sharing_parent_scope_is_not_record_covered(parent_ret):
    parent, region = _functions(parent_ret)
    book = publish_authored_ssa({parent.name: parent, region.name: region})
    layout = (17, 100, 18, 101)
    cells = tuple(
        book.post(
            OPTIONAL_VALUE_STRUCTURE, (function_scope_of(parent), payload),
            (payload, presence), stage=STRUCTURAL_CONCORDANCE,
            provenance=Derived(identity_cells(parent, payload, presence)),
            mode=Mode.CONCORD,
        )
        for payload, presence in zip(layout[0::2], layout[1::2])
    )
    publish_aggregate_return_layout(
        parent, 16, layout, sources=cells, stage=STRUCTURAL_CONCORDANCE, book=book,
    )
    assert function_scope_of(region) == function_scope_of(parent)
    assert _record_covered_return_receipts(region) == ()
    receipts = _record_covered_return_receipts(parent)
    assert [r["payload_value_id"] for r in receipts] == [17, 18]
