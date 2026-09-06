from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLContextUnavailable,
    InstalledGLSLControlShell,
    build_control_shader_artifact,
    require_gl_context,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import ControlProgram, StatementBlock
from src.compiler.deployment_stage import plan_region_deployments
from src.compiler.shader_region_pipeline import (
    ShaderRegionError,
    compile_shader_region,
    cut_shader_region,
)
from src.compiler.work_contract import PRESETS, ShaderOptimizationContract


@pytest.fixture(scope="session")
def gl():
    try:
        return require_gl_context()
    except GLContextUnavailable as error:
        pytest.skip(f"no OpenGL 4.3+ compute context: {error}")


def _elementwise_program() -> FusedProgram:
    return FusedProgram(
        version=1,
        feeds={10, 11},
        steps=[
            OpStep(0, "add", [10, 11], {}, 20),
            OpStep(1, "mul", [20], {"right_scalar": 2.0}, 21),
        ],
        outputs={"result": 21},
        meta={
            value_id: Meta(shape=(32,), dtype="float32", device="glsl")
            for value_id in (10, 11, 20, 21)
        },
    )


def _matmul_program() -> FusedProgram:
    return FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[OpStep(0, "matmul", [1, 2], {}, 3)],
        outputs={"result": 3},
        meta={
            1: Meta(shape=(8, 16), dtype="float32", device="glsl"),
            2: Meta(shape=(16, 4), dtype="float32", device="glsl"),
            3: Meta(shape=(8, 4), dtype="float32", device="glsl"),
        },
        extras={"kernel_kind": "matmul"},
    )


def _staged_program() -> CapturedFusedProgram:
    first = FusedProgram(
        1, {10, 11}, [OpStep(0, "stack", [10, 11], {"dim": 0}, 20)],
        {"first": 20}, meta={
            10: Meta((8,), "float32"), 11: Meta((8,), "float32"),
            20: Meta((2, 8), "float32"),
        }, extras={"kernel_kind": "stack"},
    )
    second = FusedProgram(
        1, {20}, [OpStep(0, "mul", [20], {"right_scalar": 2.0}, 30)],
        {"result": 30}, meta={
            20: Meta((2, 8), "float32"), 30: Meta((2, 8), "float32"),
        },
    )
    whole = FusedProgram(
        1, {10, 11}, [*first.steps, *second.steps], {"result": 30},
        meta={**first.meta, **second.meta},
    )
    return CapturedFusedProgram(whole, {}, (first, second))


def test_cut_is_a_deterministic_typed_nonrecursive_hole():
    first = cut_shader_region(7, _elementwise_program())
    second = cut_shader_region(7, _elementwise_program())
    renumbered = cut_shader_region(8, _elementwise_program())

    assert first.hole == second.hole
    assert first.capsule.capsule_digest == second.capsule.capsule_digest
    assert first.capsule.capsule_digest == renumbered.capsule.capsule_digest
    assert first.hole != renumbered.hole
    assert first.hole.marker == "__scheduled_region_7__"
    assert first.hole.invocation == "compute-dispatch"
    assert not first.hole.recursive_deployment_permitted
    assert [value.value_id for value in first.capsule.boundary.inputs] == [10, 11]
    assert first.capsule.boundary.outputs[0].metadata["shape"] == [32]
    assert first.capsule.boundary.outputs[0].metadata["dtype"] == "float32"


def test_cut_rejects_an_inner_deployment_before_shader_compilation():
    bad = _elementwise_program()
    bad.steps.insert(1, OpStep(1, "Deploy", [], {"deployment_frame": True}, 99))

    with pytest.raises(ShaderRegionError, match="recursive deployment"):
        cut_shader_region(0, bad)


def test_second_pass_assigns_shader_memory_and_blas_identity():
    artifact = compile_shader_region(cut_shader_region(3, _matmul_program()))

    step = artifact.program.steps[0]
    assert step.attrs["shader_identity"] == "glslblas_gemm"
    assert step.attrs["shader_memory_method"] == "cooperative_workgroup_tiles"
    assert {binding.storage.value for binding in artifact.memory_bindings} == {
        "storage_buffer",
    }
    assert {
        binding.promotion for binding in artifact.memory_bindings
    } == {"workgroup_shared_tile", None}
    assert artifact.tiling[0].method == "cooperative_gemm"
    assert artifact.tiling[0].tile_shape == (16, 16)
    assert "other invocations" in artifact.phases[0].barrier_after
    assert not artifact.as_record()["recursive_deployment_permitted"]


def test_second_pass_contract_can_retain_the_humble_source_algorithm():
    contract = dataclasses.replace(
        PRESETS["develop"],
        name="source-proof",
        shaders=ShaderOptimizationContract(blas_gemm="source_algorithm"),
    )
    optimized = compile_shader_region(cut_shader_region(3, _matmul_program()))
    source = compile_shader_region(
        cut_shader_region(3, _matmul_program()), contract=contract,
    )

    assert source.program.steps[0].attrs["shader_identity"] == "source_algorithm"
    assert all(
        binding.promotion is None for binding in source.memory_bindings
    )
    assert source.artifact_digest != optimized.artifact_digest

    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)), region_indices=(3,),
    )
    captured = CapturedFusedProgram(_matmul_program(), {})
    cut = cut_shader_region(3, captured)
    source_shader = build_control_shader_artifact(
        control, {3: captured}, shader_region_cuts={3: cut},
        work_contract=contract,
    )
    optimized_shader = build_control_shader_artifact(
        control, {3: captured}, shader_region_cuts={3: cut},
    )
    assert "for (uint p = 0u" in source_shader.source
    assert "shared float left_tile" not in source_shader.source
    assert "shared float left_tile" in optimized_shader.source


def test_staged_region_becomes_ordered_shader_phases_not_inner_deployments():
    captured = _staged_program()
    artifact = compile_shader_region(cut_shader_region(5, captured))

    assert len(artifact.phases) == 2
    assert "other invocations" in artifact.phases[0].barrier_after
    assert artifact.phases[1].barrier_after is None
    assert all(
        op.lower() not in {"deploy", "join"}
        for phase in artifact.phases for op in phase.operation_names
    )
    control = ControlProgram(
        StatementBlock(("__scheduled_region_5__",)), region_indices=(5,),
    )
    linked = build_control_shader_artifact(
        control,
        {5: captured},
        shader_region_cuts={5: artifact.cut},
    )
    assert len(linked.phase_sources) == 2
    assert len(linked.shader_region_links[5]["phases"]) == 2


def test_sealed_multiphase_shader_executes_as_one_outer_deployment(gl):
    captured = _staged_program()
    control = ControlProgram(
        StatementBlock(("__scheduled_region_5__",)), region_indices=(5,),
    )
    artifact = build_control_shader_artifact(control, {5: captured})
    installed = InstalledGLSLControlShell(artifact)
    left = np.arange(8, dtype=np.float32)
    right = np.arange(8, dtype=np.float32) + 10.0
    try:
        result = installed.execute({10: left, 11: right})["result"]
        np.testing.assert_allclose(
            result.numpy(), np.stack((left, right), axis=0) * 2.0,
        )
        assert installed.last_dispatches == 2
    finally:
        installed.release()


@pytest.mark.parametrize("variant", ["glslblas_gemm", "source_algorithm"])
def test_sealed_matmul_identity_executes_through_linked_artifact(gl, variant):
    program = _matmul_program()
    captured = CapturedFusedProgram(program, {})
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)), region_indices=(3,),
    )
    contract = dataclasses.replace(
        PRESETS["develop"],
        name=f"shader-{variant}",
        shaders=ShaderOptimizationContract(blas_gemm=variant),
    )
    artifact = build_control_shader_artifact(
        control, {3: captured}, work_contract=contract,
    )
    installed = InstalledGLSLControlShell(artifact)
    left = np.arange(8 * 16, dtype=np.float32).reshape(8, 16) / 17.0
    right = np.arange(16 * 4, dtype=np.float32).reshape(16, 4) / 13.0
    try:
        result = installed.execute({1: left, 2: right})["result"]
        np.testing.assert_allclose(
            result.numpy(), left @ right, rtol=2e-5, atol=2e-5,
        )
        assert installed.last_dispatches == 1
        assert artifact.shader_region_links[3]["tiling"][0]["method"] == (
            "cooperative_gemm"
            if variant == "glslblas_gemm" else "source_order_gemm"
        )
    finally:
        installed.release()


def test_deployment_cuts_and_glsl_artifact_links_the_selected_region():
    program = _elementwise_program()
    control = ControlProgram(
        StatementBlock(("__scheduled_region_4__",)),
        region_indices=(4,),
    )
    regions = {4: CapturedFusedProgram(program, {})}
    plan = plan_region_deployments(
        regions,
        control_program=control,
    )

    assert set(plan.shader_region_cuts) == {4}
    assert plan.decisions[0].choice_for("glsl").strategy == "dispatch"
    artifact = build_control_shader_artifact(
        control,
        regions,
        shader_region_cuts=plan.shader_region_cuts,
    )
    # The authoritative GLSL builder also seals automatically when an older
    # caller has not yet handed it the deployment plan explicitly.
    repeated = build_control_shader_artifact(control, regions)

    link = artifact.shader_region_links[4]
    assert link["cut"]["hole"]["marker"] == "__scheduled_region_4__"
    assert link["artifact_digest"] == repeated.shader_region_links[4][
        "artifact_digest"
    ]
    assert artifact.phase_cache_identities == repeated.phase_cache_identities
    assert "__scheduled_region_4__" not in artifact.source
    assert "float s2 = s0 + s1;" in artifact.source
