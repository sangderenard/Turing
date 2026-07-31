from llvmlite import binding as llvm

from src.common.tensors.accelerator_backends.control_ir_backends import (
    AcceleratedControlTarget,
    reduce_control_ir,
    reduce_control_ir_all_targets,
)
from src.compiler.control_source import (
    ControlProgram,
    LoopBlock,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
)


def _program():
    return ControlProgram(
        root=SequenceBlock(
            (
                LoopBlock(
                    induction="frame",
                    start="0",
                    stop="frame_count",
                    step="1",
                    body=SequenceBlock(
                        (
                            StatementBlock(("__scheduled_region_3__",)),
                            ValidationBlock(
                                predicate_value_id=91,
                                error_code=7,
                            ),
                        )
                    ),
                ),
                StateMachineTick(
                    state="codec_state",
                    cases=(
                        ("0", StatementBlock(("__scheduled_region_8__",))),
                        ("1", StatementBlock(("__scheduled_region_9__",))),
                    ),
                ),
                StreamPublishBlock(
                    stream_id=2,
                    value_id=75,
                    count_value_id=76,
                    predicate_value_id=92,
                    final=True,
                ),
            )
        ),
        region_indices=(3, 8, 9),
    )


def test_control_ir_renders_same_structure_for_c_and_glsl():
    rendered = reduce_control_ir_all_targets(_program())
    c_source = rendered[AcceleratedControlTarget.C].source
    glsl_source = rendered[AcceleratedControlTarget.GLSL].source

    assert "for (int frame = 0; frame < frame_count; frame += 1)" in c_source
    assert "switch (codec_state)" in c_source
    assert "turing_region_3();" in c_source
    assert "turing_stream_publish(2u, value_75, value_76, true)" in c_source

    assert "for (int frame = 0; frame < frame_count; frame += 1)" in glsl_source
    assert "switch (codec_state)" in glsl_source
    assert "turing_region_8();" in glsl_source
    assert glsl_source.startswith("void turing_control()")


def test_control_ir_renders_verified_llvm_blocks_and_phi():
    rendered = reduce_control_ir(
        _program(),
        AcceleratedControlTarget.LLVM_SSA,
    )

    assert "phi i64" in rendered.source
    assert "switch i64 %codec_state" in rendered.source
    assert "call void @turing_region_3()" in rendered.source
    assert "call void @turing_validation_error(i32 7)" in rendered.source
    assert "call void @turing_stream_publish(i32 2, i64 75, i64 76, i1 true)" in rendered.source

    module = llvm.parse_assembly(rendered.source)
    module.verify()


def test_target_specific_region_bodies_are_composed_without_changing_control():
    program = ControlProgram(
        root=LoopBlock(
            induction="i",
            start="0",
            stop="3",
            step="1",
            body=StatementBlock(("__scheduled_region_4__",)),
        ),
        region_indices=(4,),
    )

    c_source = reduce_control_ir(
        program,
        AcceleratedControlTarget.C,
        region_bodies={4: ("c_body();",)},
    ).source
    llvm_source = reduce_control_ir(
        program,
        AcceleratedControlTarget.LLVM_SSA,
        region_bodies={4: ("call void @turing_region_4()",)},
    ).source

    assert "c_body();" in c_source
    assert "__scheduled_region_4__" not in c_source
    assert "phi i64" in llvm_source
    llvm.parse_assembly(llvm_source).verify()
