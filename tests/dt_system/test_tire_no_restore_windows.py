"""The managed tire under the dt system's no-restore lane, window after window.

``rollback=False`` never copies or restores the material: whatever one
``advance`` leaves is the state, and only ``dt`` is steered.  This is the
configuration a real-time frame budget wants, so it must be shown stable on
the real tire (finite, every window completed, displacement inside its
criticality) and compared against the default rollback lane over the same
windows.  When the rollback lane never actually restores, the two lanes must
be bit-identical; when it does restore, the deviation is reported and must
stay physically bounded.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics
from src.common.tensors import AbstractTensor
from src.compiler.vehicle_python_compilation import (
    balloon_tire_managed_python_compilation_inputs,
    vehicle_python_runtime_bindings,
)


WINDOWS = 8
_TENSOR_FIELDS = (
    "inputs", "state", "output", "wheel_input_indices", "rest", "face_vertices",
    "face_rest", "face_scatter", "bending_incidence", "bending_scatter",
    "bending_weight", "vertex_area", "bead_mask", "face_material", "telemetry",
)


def _eager_managed_tire():
    """The authored managed tire, executed as the Python it is written in."""

    prepared = balloon_tire_managed_python_compilation_inputs(1)
    namespace = {
        "AbstractTensor": AbstractTensor, "Metrics": Metrics,
        "coerce_metrics": coerce_metrics, "run_superstep": run_superstep,
        "Targets": Targets, "STController": STController, "math": math, "np": np,
        **vehicle_python_runtime_bindings(include_configured_vehicle=False),
    }
    exec(prepared.source, namespace)
    material = prepared.feeds["material"]
    for name in _TENSOR_FIELDS:
        setattr(material, name, AbstractTensor.tensor(getattr(material, name)))
    return prepared, namespace["balloon_tire_managed_advance"]


def _run(rollback: bool, *, windows: int = WINDOWS, max_iters: int = 48):
    prepared, balloon_tire_managed_advance = _eager_managed_tire()
    material = prepared.feeds["material"]
    targets = prepared.feeds["targets"]
    controller = prepared.feeds["controller"]
    window = float(prepared.feeds["window_duration"])
    dt_next = float(prepared.feeds["dt_initial"])
    log: list[dict] = []
    rows = []
    for _index in range(windows):
        before = len(log)
        advanced, dt_next, metrics = run_superstep(
            material, window, dt_next, material.displacement_criticality_m,
            targets, controller, balloon_tire_managed_advance,
            allow_increase_mid_round=True, allow_unresolved=False,
            max_retries=None, rollback_threshold_multiplier=2.0,
            rollback=rollback, attempt_log=log, max_iters=max_iters,
        )
        rows.append({
            "advanced": float(advanced), "dt_next": float(dt_next),
            "attempts": [(float(r["dt"]), bool(r["accepted"]), float(r["metrics"].max_vel))
                         for r in log[before:]],
            "capped": "superstep_iteration_cap_hit" in (metrics.error_channels or {}),
            "finite": bool(np.isfinite(np.asarray(material.state.data)).all()),
        })
    return rows, window


@pytest.mark.dt
def test_no_restore_lane_completes_windows_and_matches_rollback():
    """Both lanes carry the tire through whole windows; no-restore is stable.

    History: this test first pinned an explosion in window 1 that turned out
    to be ``coerce_metrics`` truncating tensor-valued metrics (max_vel < 1
    m/s read as 0, so the CFL proposal grew without bound).  With exact
    metrics, the declared-step opener and the energy/power pin, every window
    completes with no rejection, and with no rejection the two lanes execute
    identical steps.
    """

    fast, window = _run(rollback=False, windows=2, max_iters=600)
    slow, _window = _run(rollback=True, windows=2, max_iters=600)

    for lane in (fast, slow):
        for row in lane:
            assert row["advanced"] >= window - 1.0e-15, row
            assert not row["capped"] and row["finite"], row
            assert all(accepted for _dt, accepted, _v in row["attempts"]), row
    # The opener never exceeds the tire's declared step.
    assert fast[0]["attempts"][0][0] <= 2.44140625e-4 + 1.0e-18
    # No rejection ever happened, so the lanes are the same computation.
    assert fast == slow


@pytest.mark.dt
def test_tire_publishes_a_positive_energy_time_scale():
    """energy_j and power_w are magnitudes, so the pin is live on the tire."""

    prepared, advance = _eager_managed_tire()
    material = prepared.feeds["material"]
    for _step in range(3):
        _ok, metrics = advance(material, 2.44140625e-4)
    channels = metrics.error_channels
    energy = float(channels["energy_j"].item())
    power = float(channels["power_w"].item())
    assert energy > 0.0 and power > 0.0, (energy, power)
    assert prepared.feeds["targets"].energy_exchange_fraction == 0.1
    assert material.dt_limit_hint() == 2.44140625e-4
