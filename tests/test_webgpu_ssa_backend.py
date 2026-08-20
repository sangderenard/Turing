from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.precompile_to_ssa import (
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
)
from src.compiler.ssa_webgpu_backend import emit_module, plan_wgsl_launch
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def _flat_artifact(source: str, feeds: dict[str, np.ndarray]):
    aot = compile_ast_aot(source, "kernel", feeds, precompile_only=True)
    program = getattr(
        aot.compiled_shell_program, "program", aot.compiled_shell_program
    )
    function, shortfalls = lower_fused_program_to_ssa(
        program, function_name="kernel"
    )
    assert not shortfalls
    returned = next(
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
    )
    count = int(next(iter(feeds.values())).size)
    return emit_module(
        IRModule({"kernel": function}),
        name="kernel",
        outputs={"kernel": returned},
        count=count,
    )


def test_ast_generated_float32_program_emits_wgsl_compute():
    artifact = _flat_artifact(
        """
def kernel(x, gain):
    return x * gain + 1.0
""",
        {
            "x": np.asarray([-2.0, 0.5, 3.0], dtype=np.float32),
            "gain": np.asarray([0.5, 2.0, 4.0], dtype=np.float32),
        },
    )

    assert artifact.complete
    assert artifact.shortfalls == ()
    assert "@compute @workgroup_size(32, 1, 1)" in artifact.source
    assert "var<storage, read>" in artifact.source
    assert "output_0[linear_index]" in artifact.source
    assert artifact.api.to_mapping()["metadata"]["dispatch_workgroups"] == (1, 1, 1)


@pytest.mark.parametrize("iterations", [3, 17])
def test_ast_generated_loop_uses_structured_wgsl(iterations):
    source = """
def recurrent(x, n):
    acc = x * 0.0
    for _ in range(n):
        acc = acc + x
    return acc


def kernel(x, n):
    return recurrent(x, n)
"""
    aot = compile_ast_aot(
        source,
        "kernel",
        {"x": np.asarray([1.0, 2.0], dtype=np.float32), "n": iterations},
        precompile_only=True,
        backend="webgpu",
        remove_loops=False,
    )
    lowered = lower_precompile_and_control_to_ssa(
        aot.compiled_shell_program,
        aot.shell_control_program,
        numerical_name="kernel",
        control_name="kernel_control",
        region_programs=aot.region_programs,
    )
    assert not lowered.shortfalls
    assert len(lowered.cycles) == 1
    control = lowered.module.functions["kernel_control"]
    carried = next(
        instruction.res
        for instruction in control.blocks["loop_header"].instrs
        if instruction.attributes.get("binding") == "loop_carried"
    )

    artifact = emit_module(
        lowered.module,
        name="kernel",
        outputs={"kernel_control": (carried,)},
        count=2,
    )

    assert artifact.complete
    assert f": i32 = {iterations}i;" in artifact.source
    assert "fn numerical_region_0(" in artifact.source
    assert "fn numerical_region_1(" in artifact.source
    assert "  loop {" in artifact.source
    assert "    continuing {" in artifact.source
    assert "break if (!" in artifact.source


def test_ast_generated_bitwise_operation_uses_wgsl_bitcasts():
    program = FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[OpStep(0, "bitand", [1, 2], {}, 3)],
        outputs={"result": 3},
    )
    function, shortfalls = lower_fused_program_to_ssa(
        program, function_name="kernel"
    )
    assert not shortfalls
    returned = next(
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
    )
    artifact = emit_module(
        IRModule({"kernel": function}),
        name="kernel",
        outputs={"kernel": returned},
        count=2,
    )

    assert artifact.complete
    assert "bitcast<u32>" in artifact.source
    assert "bitcast<f32>" in artifact.source


def test_float64_is_a_named_webgpu_core_shortfall():
    artifact = _flat_artifact(
        """
def kernel(x):
    return x + 1.0
""",
        {"x": np.asarray([1.0, 2.0], dtype=np.float64)},
    )

    assert not artifact.complete
    assert any(
        "float64 has no WebGPU core equivalent" in item.reason
        for item in artifact.shortfalls
    )


def test_canonical_ssa_diamond_emits_structured_if_else():
    condition = SSAValue(0, "bool")
    left = SSAValue(1, "float32")
    right = SSAValue(2, "float32")
    passed = SSAValue(3, "float32")
    failed = SSAValue(4, "float32")
    merged = SSAValue(5, "float32")
    function = Function("choose", [condition, left, right], {
        "entry": BasicBlock("entry", [
            Instr(
                "CondBr", [condition], None,
                attributes={"true_target": "passed", "false_target": "failed"},
            ),
        ]),
        "passed": BasicBlock("passed", [
            Instr("Add", [left, right], passed),
            Instr("Br", [], None, attributes={"target": "merge"}),
        ]),
        "failed": BasicBlock("failed", [
            Instr("Sub", [left, right], failed),
            Instr("Br", [], None, attributes={"target": "merge"}),
        ]),
        "merge": BasicBlock("merge", [
            Instr(
                "Phi", [passed, failed], merged,
                attributes={"incoming_blocks": ("passed", "failed")},
            ),
            Instr("Ret", [merged], None),
        ]),
    })

    artifact = emit_module(
        IRModule({"choose": function}),
        name="choose",
        outputs={"choose": (merged,)},
    )

    assert artifact.complete
    assert "if (v_0) {" in artifact.source
    assert "} else {" in artifact.source
    assert "v_5 = v_3;" in artifact.source
    assert "v_5 = v_4;" in artifact.source


def test_launch_planning_obeys_webgpu_minimum_limits():
    launch = plan_wgsl_launch(256 * 65535 + 1)

    assert launch.workgroup_size == (256, 1, 1)
    assert launch.groups == (65535, 2, 1)
    assert launch.limits.max_invocations_per_workgroup == 256
    assert launch.limits.max_workgroups_per_dimension == 65535
    assert launch.deployment.backend == "webgpu"
    assert launch.deployment.compute.groups == launch.groups
