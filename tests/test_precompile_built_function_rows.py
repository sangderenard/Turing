"""A precompile-built numerical function is on the book at creation."""

from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.concordance_declarations import SSA_VALUE, SSAValueFact
from src.compiler.identity_concordance import current_identity_book
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.structural_concordance import function_value_structure


def _program():
    steps = [
        OpStep(0, "neg", [0], {}, 1),
        OpStep(1, "add", [1], {"right_scalar": 1.0}, 2),
    ]
    return FusedProgram(
        version=1,
        feeds={0},
        steps=steps,
        outputs={"result": 2},
        meta={i: Meta((4,), "float32", "glsl") for i in (0, 1, 2)},
    )


def test_numerical_function_values_have_ssa_value_rows_at_creation():
    function, _ = lower_fused_program_to_ssa(
        _program(), function_name="numerical_precompile_rows_probe",
    )
    book = current_identity_book()
    ids = {int(value.id) for value in function.args} | {
        int(instruction.res.id)
        for instruction in function.blocks["entry"].instrs
        if instruction.res is not None
    }
    assert ids == {0, 1, 2}
    for value_id in ids:
        ref = book.latest_ref(SSA_VALUE, (function.name, value_id))
        assert ref is not None, value_id
        assert isinstance(book.page(SSA_VALUE).cell(ref.row, ref.column), SSAValueFact)
        structure, _cell = function_value_structure(function, value_id, book=book)
        assert structure.shape == (4,)
