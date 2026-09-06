"""Deployment lowering profiles and strategy selection."""

from __future__ import annotations

import pytest

from src.compiler.deployment_lowering import (
    ComputeDispatchLimits,
    DISPATCH,
    POOL,
    SERIAL,
    DeploymentLoweringProfile,
    deployment_profile,
    select_deployment_strategy,
)


def test_serial_is_mandatory_in_every_profile():
    with pytest.raises(ValueError, match="serial"):
        DeploymentLoweringProfile(backend="x", strategies=(POOL,))


def test_unknown_backend_degrades_to_serial_only():
    profile = deployment_profile("some-future-backend")
    assert profile.strategies == (SERIAL,)
    assert "no declared deployment profile" in profile.note


def test_thread_worker_class_selects_pool_where_declared():
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers", join_mode="barrier",
    )
    assert choice.strategy == POOL
    assert choice.parallel
    assert any("turing_pool.c" in reason for reason in choice.reasons)


def test_shader_class_selects_dispatch_on_gpu_backends():
    choice = select_deployment_strategy(
        backend="webgpu", execution_class="shader-compute",
    )
    assert choice.strategy == DISPATCH


def test_shader_choice_owns_device_valid_compute_geometry():
    limits = ComputeDispatchLimits(
        max_group_count=(4, 3, 2),
        max_group_size=(128, 8, 4),
        max_invocations=128,
    )
    for backend in ("glsl", "webgpu"):
        choice = select_deployment_strategy(
            backend=backend,
            execution_class="shader-compute",
            work=1000,
            preferred_local_size=128,
            compute_limits=limits,
        )
        assert choice.strategy == DISPATCH
        assert choice.compute is not None
        assert choice.compute.workgroup_size == (128, 1, 1)
        assert choice.compute.groups == (4, 2, 1)
        assert any("compute geometry chosen" in item for item in choice.reasons)


def test_shader_geometry_preserves_flat_identity_when_grid_folds():
    limits = ComputeDispatchLimits(
        max_group_count=(2, 2, 2),
        max_group_size=(64, 1, 1),
        max_invocations=64,
    )
    choice = select_deployment_strategy(
        backend="webgpu", execution_class="shader-compute", work=500,
        compute_limits=limits, preferred_local_size=64,
    )
    assert choice.compute is not None
    assert choice.compute.groups == (2, 2, 2)
    assert 500 <= 2 * 2 * 2 * 64


def test_capability_gap_degrades_to_serial_with_reasons():
    # Fortran declares no parallel strategy today.
    choice = select_deployment_strategy(
        backend="fortran", execution_class="thread-workers",
    )
    assert choice.strategy == SERIAL
    assert not choice.parallel
    assert any("does not declare" in reason for reason in choice.reasons)


def test_unsupported_join_mode_degrades_visibly():
    # The C pool only honors barrier joins in parallel.
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers", join_mode="reduce",
    )
    assert choice.strategy == SERIAL
    assert any("join mode 'reduce'" in reason for reason in choice.reasons)


def test_unknown_execution_class_is_reported_and_serial():
    choice = select_deployment_strategy(
        backend="llvm", execution_class="not-a-class",
    )
    assert choice.strategy == SERIAL
    assert any("unknown execution class" in reason for reason in choice.reasons)


# ---------------------------------------------------------------------------
# Strategic tiling: chunk geometry, core-stated budgets, nested tempering.
# All evidence is optional and inert when absent -- the assertions above
# this section double as the proof that no-evidence behavior is unchanged.
# ---------------------------------------------------------------------------

from src.compiler.deployment_calibration import (  # noqa: E402
    CalibrationVerdict,
    WorkloadSignature,
    machine_fingerprint,
)


def _pool_verdict(workers: int = 4, speedup: float = 2.5) -> CalibrationVerdict:
    return CalibrationVerdict(
        signature=WorkloadSignature(
            backend="c", kind="region", work=100_000, identity="t1",
        ),
        machine=machine_fingerprint(),
        best_strategy="pool",
        best_workers=workers,
        speedup=speedup,
        serial_seconds=1.0,
        best_seconds=1.0 / speedup,
        samples=3,
    )


def test_a_measured_pool_choice_carries_a_strategic_chunk():
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        calibration=_pool_verdict(workers=4), work=100_000,
    )
    assert choice.strategy == POOL
    assert choice.workers == 4
    # 100_000 work over 4 workers at 4 claims each.
    assert choice.chunk == 100_000 // 16
    assert any("chunk" in reason for reason in choice.reasons)


def test_without_work_evidence_the_chunk_stays_executor_default():
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        calibration=_pool_verdict(workers=4),
    )
    assert choice.strategy == POOL
    assert choice.chunk is None


def test_stated_cores_supply_a_budget_only_without_measurement():
    unmeasured = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        cores=8, work=64_000,
    )
    assert unmeasured.strategy == POOL
    assert unmeasured.workers == 7
    assert unmeasured.chunk == 64_000 // 28
    assert any(
        "caller as one execution slot" in reason
        for reason in unmeasured.reasons
    )
    measured = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        calibration=_pool_verdict(workers=2), cores=8, work=64_000,
    )
    # Measurement outranks the stated core count.
    assert measured.workers == 2


def test_browser_pool_workers_do_not_use_cpu_caller_accounting():
    choice = select_deployment_strategy(
        backend="wasm", execution_class="thread-workers",
        cores=4, work=64_000,
    )
    assert choice.strategy == POOL
    assert choice.workers == 4
    assert not any("caller as one execution slot" in reason
                   for reason in choice.reasons)


def test_nesting_tempers_the_worker_budget_with_a_reason():
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        cores=8, work=64_000, nesting_depth=1,
    )
    assert choice.strategy == POOL
    assert choice.workers == 3
    assert any("tempered" in reason for reason in choice.reasons)


def test_a_budget_tempered_to_one_worker_goes_serial():
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        cores=2, nesting_depth=3, work=64_000,
    )
    assert choice.strategy == SERIAL
    assert any("pure overhead" in reason for reason in choice.reasons)


def test_no_evidence_at_all_changes_nothing():
    bare = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
    )
    assert bare.strategy == POOL
    assert bare.workers is None
    assert bare.chunk is None
