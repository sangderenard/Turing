"""Execution-class assignment for scheduled regions.

Exercises the per-region planner in ``deployment_classification`` against
synthetic region ``FusedProgram``s and real registry capabilities: shader
eligibility comes from the actual compute-stage ``machine_targets`` entries,
not a mocked vocabulary, so a registry change that would silently strand a
class shows up here.
"""

from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler import machine_targets
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
)
from src.compiler.deployment_classification import (
    GRAPHICS_OUTPUT,
    HOST_LINEAR,
    SHADER_COMPUTE,
    THREAD_WORKERS,
    classify_region_executions,
)
from src.compiler.wasm_class_modules import COLLECTIVE_FUSED_OPERATIONS


def _compute_stage_op() -> str:
    """A pointwise operation every compute-stage target really supports."""

    vocabularies = [
        target.capabilities.supported_operations
        for target in machine_targets.targets().values()
        if target.capabilities.stage == "compute"
        and not target.capabilities.deprecated
        and target.capabilities.supported_operations is not None
    ]
    assert vocabularies, "no compute-stage machine target is registered"
    shared = frozenset.intersection(*vocabularies) - COLLECTIVE_FUSED_OPERATIONS
    assert shared, "compute-stage targets share no pointwise operation"
    return sorted(shared)[0]


def _region(op_name: str, outputs: dict[str, int]) -> FusedProgram:
    return FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(step_id=0, op_name=op_name, input_ids=[1], result_id=2)],
        outputs=dict(outputs),
    )


def _barrier_deployment(*lane_regions: tuple[int, ...]) -> ControlDeploymentRegion:
    return ControlDeploymentRegion(
        region_id=0,
        kind="parallel_candidate",
        schedule="independent_lanes",
        lanes=tuple(
            ControlDeploymentLane(index=index, region_indices=regions)
            for index, regions in enumerate(lane_regions)
        ),
    )


def test_shader_capable_region_classifies_as_shader_compute():
    classified = classify_region_executions(
        {0: _region(_compute_stage_op(), {"result": 2})},
    )
    record = classified[0]
    assert record.execution_class == SHADER_COMPUTE
    assert SHADER_COMPUTE in record.eligible
    assert record.compute_shader_targets
    assert record.extent_effect == "pointwise"


def test_inexpressible_region_falls_back_to_host_linear():
    classified = classify_region_executions(
        {3: _region("definitely_not_a_registered_op", {"result": 2})},
    )
    record = classified[3]
    assert record.execution_class == HOST_LINEAR
    assert record.eligible == ()
    assert not record.compute_shader_targets


def test_barrier_lane_membership_classifies_as_thread_workers():
    deployment = _barrier_deployment((0,), (1,))
    classified = classify_region_executions(
        {
            0: _region("definitely_not_a_registered_op", {"a": 2}),
            1: _region("definitely_not_a_registered_op", {"b": 2}),
        },
        deployment_regions=(deployment,),
    )
    for index in (0, 1):
        record = classified[index]
        assert record.execution_class == THREAD_WORKERS
        assert record.deployment_region_id == 0
        assert record.lane_count == 2


def test_collective_extent_effect_vetoes_thread_workers():
    deployment = _barrier_deployment((0,), (1,))
    classified = classify_region_executions(
        {
            0: _region("sum", {"a": 2}),
            1: _region("sum", {"b": 2}),
        },
        deployment_regions=(deployment,),
    )
    record = classified[0]
    assert record.extent_effect == "collective"
    assert THREAD_WORKERS not in record.eligible
    assert any("vetoed" in reason for reason in record.reasons)


def test_single_lane_deployment_is_not_parallelism():
    deployment = _barrier_deployment((0,))
    classified = classify_region_executions(
        {0: _region("definitely_not_a_registered_op", {"a": 2})},
        deployment_regions=(deployment,),
    )
    assert classified[0].execution_class == HOST_LINEAR
    assert classified[0].lane_count == 0


def test_presentation_channels_take_precedence():
    classified = classify_region_executions(
        {0: _region(_compute_stage_op(), {"red": 2, "green": 2, "blue": 2})},
    )
    record = classified[0]
    assert record.execution_class == GRAPHICS_OUTPUT
    assert record.presentation_outputs == ("blue", "green", "red")
    # The eligibility set keeps the fallback: a page whose surface cannot
    # read this placement still knows the region is shader-expressible.
    assert SHADER_COMPUTE in record.eligible


def test_configured_channels_override_the_default_trio():
    classified = classify_region_executions(
        {0: _region(_compute_stage_op(), {"luminance": 2})},
        presentation_channels=frozenset({"luminance"}),
    )
    assert classified[0].execution_class == GRAPHICS_OUTPUT


def test_records_serialize_for_the_bundle_manifest():
    record = classify_region_executions(
        {0: _region(_compute_stage_op(), {"red": 2})},
    )[0].as_record()
    assert record["execution_class"] == GRAPHICS_OUTPUT
    assert isinstance(record["eligible"], list)
    assert isinstance(record["reasons"], list)
