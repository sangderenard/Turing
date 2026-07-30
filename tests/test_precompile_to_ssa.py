from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import (
    ControlProgram,
    ControlUniform,
    LoopBlock,
    SequenceBlock,
    StatementBlock,
)
from src.compiler.precompile_to_ssa import (
    find_ssa_cycles,
    lower_control_program_to_ssa,
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
)


def _program(*steps):
    value_ids = {0}
    value_ids.update(step.result_id for step in steps)
    return FusedProgram(
        version=1,
        feeds={0},
        steps=list(steps),
        outputs={"result": steps[-1].result_id},
        meta={
            value_id: Meta((4,), "float32", "glsl")
            for value_id in value_ids
        },
    )


def test_numerical_lowering_uses_existing_ssa_names_without_rewriting():
    program = _program(
        OpStep(0, "neg", [0], {}, 1),
        OpStep(1, "add", [1], {"right_scalar": 1.0}, 2),
    )

    function, shortfalls = lower_fused_program_to_ssa(program)

    assert shortfalls == ()
    assert [instruction.op for instruction in function.blocks["entry"].instrs] == [
        "Neg",
        "Add",
        "Ret",
    ]
    assert function.blocks["entry"].instrs[0].res.id == 1
    assert function.blocks["entry"].instrs[1].attributes == {
        "right_scalar": 1.0
    }


def test_numerical_lowering_names_unsupported_op_and_dependent_output():
    program = _program(
        OpStep(0, "sin", [0], {}, 1),
        OpStep(1, "add", [1], {"right_scalar": 1.0}, 2),
    )

    function, shortfalls = lower_fused_program_to_ssa(program)

    assert [item.name for item in shortfalls] == [
        "sin",
        "add",
        "output",
    ]
    assert [instruction.op for instruction in function.blocks["entry"].instrs] == [
        "Ret"
    ]


def test_planner_loop_becomes_phi_cfg_cycle_with_region_call():
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "frame_count",
            "1",
            StatementBlock(("__scheduled_region_7__",)),
        ),
        region_indices=(7,),
        uniforms=(ControlUniform("frame_count", 40, "int"),),
    )

    function, shortfalls = lower_control_program_to_ssa(
        control,
        first_value_id=100,
    )
    cycles = find_ssa_cycles(function)

    assert shortfalls == ()
    assert len(cycles) == 1
    assert cycles[0].represented_by_phi
    assert cycles[0].phi_blocks == ("loop_header",)
    assert [
        instruction.op
        for instruction in function.blocks["loop_header"].instrs
    ] == ["Phi", "Lt", "CondBr"]
    assert function.blocks["loop_body"].instrs[0].op == "Call"
    assert function.blocks["loop_body"].instrs[0].attributes[
        "region_index"
    ] == 7


def test_combined_lowering_retains_sequence_order_and_cycle_report():
    program = _program(OpStep(0, "mul", [0], {"right_scalar": 2}, 1))
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            LoopBlock(
                "i",
                "0",
                "count",
                "1",
                StatementBlock(("__scheduled_region_1__",)),
            ),
        )),
        region_indices=(0, 1),
        uniforms=(ControlUniform("count", 9, "int"),),
    )

    result = lower_precompile_and_control_to_ssa(program, control)

    assert set(result.module.functions) == {
        "numerical_precompile",
        "planned_control",
    }
    assert len(result.cycles) == 1
    assert result.cycles[0].represented_by_phi
    assert result.shortfalls == ()
