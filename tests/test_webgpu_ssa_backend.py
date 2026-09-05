from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.precompile_to_ssa import (
    lower_fused_program_to_ssa,
    lower_precompile_and_control_to_ssa,
)
from src.compiler.ssa_webgpu_backend import (
    benchmarkable_tensor_operations,
    emit_gemm_module,
    emit_module,
    emit_operator_module,
    plan_gemm_matrix_deployment,
    plan_wgsl_launch,
)
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


def test_canonical_sympy_pi_and_tanh_spell_as_webgpu_intrinsics():
    source = SSAValue(900, "float32")
    pi = SSAValue(901, "float32")
    result = SSAValue(902, "float32")
    function = Function("canonical_intrinsics", [source], {
        "entry": BasicBlock("entry", [
            Instr("Pi", [], pi, {"constant_identity": "pi"}),
            Instr("Tanh", [source], result),
            Instr("Ret", [pi, result], None),
        ]),
    })
    artifact = emit_module(
        IRModule({function.name: function}), name=function.name,
        outputs={function.name: (pi, result)}, count=1,
    )

    assert artifact.complete
    assert "3.14159265358979323846f" in artifact.source
    assert "tanh(v_900)" in artifact.source


def test_same_typed_outputs_can_publish_one_component_major_gpu_span():
    left = SSAValue(700, "float32")
    first = SSAValue(701, "float32")
    second = SSAValue(702, "float32")
    function = Function("packed", [left], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, left], first),
            Instr("Mul", [left, left], second),
            Instr("Ret", [first, second], None),
        ]),
    })
    artifact = emit_module(
        IRModule({"packed": function}), name="packed",
        outputs={"packed": (first, second)}, count=4,
        preferred_local_size=4, packed_outputs=True,
    )

    assert artifact.complete
    assert artifact.source.count("var<storage, read_write>") == 1
    assert "outputs[0u + linear_index]" in artifact.source
    assert "outputs[4u + linear_index]" in artifact.source
    metadata = artifact.api.to_mapping()["metadata"]
    assert metadata["packed_outputs"] is True
    assert metadata["output_span"] == [701, 702]


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


def test_fast_contract_legalizes_float64_through_a_backend_identity():
    from src.compiler.work_contract import set_active_contract

    left = SSAValue(700, "float64")
    right = SSAValue(701, "float64")
    result = SSAValue(702, "float64")
    function = Function("add", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Add", [left, right], result),
            Instr("Ret", [result], None),
        ]),
    })
    set_active_contract("fast")
    try:
        artifact = emit_module(
            IRModule({"add": function}),
            name="add",
            outputs={"add": (result,)},
            count=8,
        )
    finally:
        set_active_contract(None)

    assert artifact.complete
    assert "array<f32>" in artifact.source
    decision = artifact.backend_identity_decisions[0]
    assert decision.identity == "shader_float64_storage_to_float32"
    assert decision.applied
    assert artifact.api.to_mapping()["metadata"]["backend_identities"][0][
        "applied"
    ]


def test_repository_spellings_for_sqrt_and_extrema_emit_to_wgsl():
    value = SSAValue(710, "float32")
    other = SSAValue(711, "float32")
    root = SSAValue(712, "float32")
    maximum = SSAValue(713, "float32")
    minimum = SSAValue(714, "float32")
    function = Function("surface", [value, other], {
        "entry": BasicBlock("entry", [
            Instr("Sqrt", [value], root),
            Instr("Max", [root, other], maximum),
            Instr("Min", [maximum, value], minimum),
            Instr("Ret", [minimum], None),
        ]),
    })

    artifact = emit_module(
        IRModule({"surface": function}),
        name="surface",
        outputs={"surface": (minimum,)},
    )

    assert artifact.complete
    assert "sqrt(v_710)" in artifact.source
    assert "max(v_712, v_711)" in artifact.source
    assert "min(v_713, v_710)" in artifact.source


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


def test_webgpu_reads_the_prebaked_matrix_without_rewriting_it():
    from src.compiler.tiling_strategy import (
        build_gemm_tile_plan,
        prebake_gemm_launch_matrix,
    )

    matrix = prebake_gemm_launch_matrix(
        build_gemm_tile_plan(192, 128, 64, 64, worker_budget=7),
        variant_key="one-universal-gemm", parameter_ids={},
        total_layout={}, core_layout={}, chunk_size=1,
    )
    interpreted = plan_gemm_matrix_deployment(matrix)
    assert interpreted.module_key == "one-universal-gemm"
    assert interpreted.lane_count == 6
    assert interpreted.calls_per_lane == (1,) * 6
    assert interpreted.choice.compute.count == 6


def test_every_advertised_benchmark_operation_is_complete_wgsl():
    vocabulary = benchmarkable_tensor_operations()
    assert {"add", "mul", "sqrt", "logical_and", "bitxor"} <= vocabulary.keys()

    for operation in vocabulary:
        artifact = emit_operator_module(operation, 256)
        assert artifact.complete, (operation, artifact.shortfalls)
        assert "@compute" in artifact.source
        assert "output_0[linear_index]" in artifact.source
    logical = emit_operator_module("logical_and", 256).source
    assert "select(0.0f, 1.0f" in logical
    assert "bool(" not in logical


def test_webgpu_gemm_variants_share_the_role_and_abi_but_change_topology():
    source = emit_gemm_module(65, 33, 17, variant="source_algorithm")
    tiled = emit_gemm_module(65, 33, 17, variant="webgpu_tiled_gemm")
    source_meta = source.api.to_mapping()["metadata"]
    tiled_meta = tiled.api.to_mapping()["metadata"]

    assert source.complete and tiled.complete
    assert source_meta["role"] == tiled_meta["role"] == "blas.gemm"
    assert source_meta["role_source_sha256"] == tiled_meta["role_source_sha256"]
    assert source_meta["io_layout"] == tiled_meta["io_layout"]
    assert source_meta["io_layout"]["uniforms"][0]["name"] == "gemm_scalars"
    assert "gemm_scalars.alpha * sum" in source.source
    assert "gemm_scalars.beta * output_C" in tiled.source
    assert "for (var p = 0u" in source.source
    assert "var<workgroup> tile_A" in tiled.source
    assert "workgroupBarrier()" in tiled.source
    assert tiled.launch_plan.workgroup_size == (16, 16, 1)
    assert tiled.launch_plan.groups == (3, 5, 1)
    # Both implementations are selected WebGPU faux intrinsics.  The variant
    # chooses which custom WGSL body that intrinsic emits; it does not bypass
    # the backend-location swap for the source-shaped implementation.
    assert source_meta["backend_identities"][0]["applied"]
    assert tiled_meta["backend_identities"][0]["applied"]
    assert source_meta["variant"] == "source_algorithm"
    assert tiled_meta["variant"] == "webgpu_tiled_gemm"
    assert source_meta["backend_intrinsic"]["location"] == (
        "src.compiler.ssa_webgpu_backend:webgpublas_gemm"
    )
    assert tiled_meta["backend_intrinsic"] == source_meta["backend_intrinsic"] | {
        "shader_variant": "webgpu_tiled_gemm",
    }


def test_every_authored_blas_method_has_a_source_topology_wgsl_module():
    from src.compiler.ssa_webgpu_backend import emit_blas_module

    modules = {
        name: emit_blas_module(name, m=13, n=17, k=19)
        for name in ("scal", "axpy", "dot", "gemv", "gemm", "rot")
    }
    assert all(module.complete for module in modules.values())
    assert "scalars.alpha * input_x[index]" in modules["scal"].source
    assert "+ output_y[index]" in modules["axpy"].source
    assert "for (var i = 0u; i < 17u" in modules["dot"].source
    assert "input_A[index * 17u + column]" in modules["gemv"].source
    assert "var<workgroup> tile_A" in modules["gemm"].source
    assert "let xi = output_x[index]" in modules["rot"].source
    for name, module in modules.items():
        assert module.api.metadata["role"] == f"blas.{name}"
