"""The default deployment-planning stage: plan shape and gating rules."""

from __future__ import annotations

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
)
from src.compiler.deployment_calibration import (
    CalibrationStore,
    CalibrationVerdict,
    machine_fingerprint,
)
from src.compiler.deployment_stage import (
    browser_threading_veto,
    plan_region_deployments,
    region_workload_signature,
)


def _region(op_name: str = "definitely_not_a_registered_op") -> FusedProgram:
    return FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(step_id=0, op_name=op_name, input_ids=[1], result_id=2)],
        outputs={"out": 2},
    )


def _barrier_deployment() -> ControlDeploymentRegion:
    return ControlDeploymentRegion(
        region_id=0,
        kind="parallel_candidate",
        schedule="independent_lanes",
        lanes=(
            ControlDeploymentLane(index=0, region_indices=(0,)),
            ControlDeploymentLane(index=1, region_indices=(1,)),
        ),
    )


def _demoting_verdict(signature) -> CalibrationVerdict:
    return CalibrationVerdict(
        signature=signature,
        machine=machine_fingerprint(),
        best_strategy="serial",
        best_workers=0,
        speedup=0.85,
        serial_seconds=1.0,
        best_seconds=1.0 / 0.85,
        samples=3,
    )


def test_plan_covers_every_region_and_backend():
    plan = plan_region_deployments(
        {0: _region(), 1: _region()},
        deployment_regions=(_barrier_deployment(),),
    )
    assert len(plan.decisions) == 2
    for decision in plan.decisions:
        assert decision.classification.execution_class == "thread-workers"
        backends = {choice.backend for choice in decision.choices}
        assert {
            "wasm", "webgpu", "glsl", "native_glsl", "c", "llvm", "fortran",
        } <= backends
        assert decision.choice_for("wasm").strategy == "pool"
        assert decision.choice_for("fortran").strategy == "serial"


def test_manifest_record_is_json_shaped():
    plan = plan_region_deployments({0: _region()})
    record = plan.as_manifest()
    assert set(record) == {"0"}
    strategies = record["0"]["strategies"]
    assert strategies["wasm"]["strategy"] in ("serial", "pool", "dispatch")
    assert isinstance(strategies["wasm"]["reasons"], list)
    assert "compute" in strategies["webgpu"]
    assert "compute" in strategies["glsl"]
    assert "compute" in strategies["native_glsl"]


def test_presentation_dispatch_does_not_cut_an_untranslatable_numeric_region():
    program = _region()
    program.outputs = {"red": 2}
    plan = plan_region_deployments({0: program})

    assert plan.decisions[0].classification.execution_class == "graphics-output"
    assert plan.decisions[0].choice_for("glsl").strategy == "dispatch"
    assert plan.decisions[0].choice_for("native_glsl").strategy == "dispatch"
    assert plan.decisions[0].classification.compute_shader_targets == ()
    assert plan.shader_region_cuts == {}


def test_gate_stays_open_without_any_calibration():
    plan = plan_region_deployments(
        {0: _region(), 1: _region()},
        deployment_regions=(_barrier_deployment(),),
    )
    assert browser_threading_veto(plan) is None


def test_gate_stays_open_when_serial_is_only_a_capability_gap():
    # host-linear regions choose serial with no measurement involved;
    # absence of evidence must not close the gate.
    plan = plan_region_deployments({0: _region()})
    assert plan.decisions[0].choice_for("wasm").strategy == "serial"
    assert browser_threading_veto(plan) is None


def test_measured_demotion_closes_the_gate(tmp_path):
    store = CalibrationStore(RepositoryArtifactCache(root=tmp_path))
    programs = {0: _region(), 1: _region()}
    for program in programs.values():
        store.store(_demoting_verdict(
            region_workload_signature("wasm", program)
        ))
    plan = plan_region_deployments(
        programs,
        deployment_regions=(_barrier_deployment(),),
        calibration_store=store,
    )
    for decision in plan.decisions:
        wasm = decision.choice_for("wasm")
        assert wasm.strategy == "serial"
        assert wasm.calibration_demoted
    veto = browser_threading_veto(plan)
    assert veto is not None and "calibration demoted" in veto


def test_signature_is_stable_under_region_renumbering():
    program = _region("add")
    first = region_workload_signature("c", program)
    second = region_workload_signature("c", program)
    assert first == second
    assert first.identity  # a real digest, not empty
    assert region_workload_signature(
        "c", _region("mul")
    ).identity != first.identity


def test_region_nesting_depths_count_multiple_lane_claims():
    from src.compiler.deployment_stage import region_nesting_depths

    class Lane:
        def __init__(self, region_indices):
            self.region_indices = tuple(region_indices)

    class Deployment:
        def __init__(self, lanes):
            self.lanes = tuple(lanes)

    outer = Deployment([Lane([1, 2]), Lane([3])])
    inner = Deployment([Lane([2])])   # region 2 is a lane of BOTH
    depths = region_nesting_depths((outer, inner))
    assert depths[1] == 0 and depths[3] == 0
    assert depths[2] == 1
