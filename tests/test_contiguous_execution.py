from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.contiguous_execution import IndexRelation, contiguate


def _program(op, inputs, output, shapes, attrs=None):
    return FusedProgram(
        version=1,
        feeds=set(inputs),
        steps=[OpStep(0, op, list(inputs), attrs or {}, output)],
        outputs={"result": output},
        meta={
            value_id: Meta(shape, "float32", "glsl")
            for value_id, shape in shapes.items()
        },
    )


def test_contiguate_keeps_scalar_broadcast_in_same_shader_phase():
    programs = (
        _program("add", (1, 2), 3, {1: (128,), 2: (1,), 3: (128,)}),
        _program("mul", (3, 4), 5, {3: (128,), 4: (1,), 5: (128,)}),
    )
    plan = contiguate(programs)
    assert plan.is_single_dispatch
    assert all(
        operation.relation is IndexRelation.SAME_INDEX
        for operation in plan.phases[0].operations
    )


def test_contiguate_exposes_stack_as_a_real_dispatch_phase():
    programs = (
        _program("add", (1, 2), 3, {1: (128,), 2: (1,), 3: (128,)}),
        _program(
            "stack",
            (3, 4, 5),
            6,
            {3: (128,), 4: (128,), 5: (128,), 6: (128, 3)},
            {"dim": -1},
        ),
        _program("mul", (6, 7), 8, {6: (128, 3), 7: (1,), 8: (128, 3)}),
    )
    plan = contiguate(programs)
    assert plan.dispatch_count == 3
    assert plan.phases[1].operations[0].relation is (
        IndexRelation.CROSS_INVOCATION
    )
    assert "other invocations" in plan.phases[1].barrier_after
    assert "other invocations" in plan.phases[2].barrier_before


def test_contiguate_exposes_computed_scalar_before_wide_broadcast():
    programs = (
        _program("add", (1, 2), 3, {1: (1,), 2: (1,), 3: (1,)}),
        _program("mul", (3, 4), 5, {3: (1,), 4: (128,), 5: (128,)}),
    )
    plan = contiguate(programs)
    assert plan.dispatch_count == 2
    assert plan.phases[0].operations[0].relation is (
        IndexRelation.SAME_INDEX
    )
    assert "scalar" in plan.phases[0].barrier_after
