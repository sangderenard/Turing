"""Deployment lowering profiles and strategy selection."""

from __future__ import annotations

import pytest

from src.compiler.deployment_lowering import (
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
