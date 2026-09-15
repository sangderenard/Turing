import pytest

from src.compiler.control_source import (
    ControlProgram, ControlUniform, LoopBlock, SequenceBlock,
)
from src.compiler.precompile_to_ssa import lower_control_sections_to_ssa


@pytest.mark.parametrize("dtype", ["float64", "int64", "int32"])
def test_loop_uniform_preserves_explicit_storage_contract(dtype):
    control = ControlProgram(
        LoopBlock("i", "0", "count", "1", SequenceBlock(())), (),
        uniforms=(ControlUniform("count", 40, "int"),),
    )
    module, shortfalls, _ = lower_control_sections_to_ssa(
        control, identity_table={"count": (40,)},
        function_parameters=("count",), value_dtypes={40: dtype},
    )
    assert not shortfalls
    function = module.functions["planned_control"]
    count = next(value for value in function.args if value.id == 40)
    assert count.dtype == dtype
    comparison = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs if instruction.op == "Lt"
    )
    assert comparison.args[1] is count
