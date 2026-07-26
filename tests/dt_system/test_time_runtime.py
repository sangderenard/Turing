from __future__ import annotations

import math

import pytest

from src.common import (
    ManagedTimeRuntime,
    TimeWindowRequest,
)
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_scaler import Metrics


class ManagedState:
    def __init__(self) -> None:
        self.time = 0.0

    def copy_shallow(self):
        return self.time

    def restore(self, snapshot) -> None:
        self.time = float(snapshot)


def test_managed_runtime_maps_absolute_events_and_reports_reruns():
    state = ManagedState()

    def advance(value, dt):
        value.time += float(dt)
        return True, Metrics(
            1.0,
            0.0,
            0.0,
            0.0,
            error_channels={"field_residual": float(dt)},
        )

    runtime = ManagedTimeRuntime(
        state,
        advance,
        dx=1.0,
        targets=Targets(
            1.0,
            1.0,
            1.0,
            error_limits={"field_residual": 0.1},
        ),
        controller=STController(dt_min=1.0e-9),
        generation=4,
        initial_time=10.0,
    )
    request = TimeWindowRequest(
        request_id=12,
        generation=4,
        t_start=10.0,
        t_end=10.3,
        dt_initial=0.2,
        event_times=(10.15, 10.25),
    )

    report = runtime.advance(request)

    assert report.exact_landing
    assert runtime.current_time == pytest.approx(10.3)
    assert state.time == pytest.approx(0.3)
    assert report.result.rejected_attempts >= 1
    assert report.result.landed_boundaries == pytest.approx((0.15, 0.25))


def test_managed_runtime_rolls_back_entire_failed_window():
    state = ManagedState()

    def advance(value, dt):
        value.time += float(dt)
        return True, Metrics(
            0.0,
            0.0,
            0.0,
            0.0,
            hard_failure=value.time > 0.1,
        )

    controller = STController(dt_min=1.0e-9)
    runtime = ManagedTimeRuntime(
        state,
        advance,
        dx=1.0,
        targets=Targets(1.0, 1.0, 1.0),
        controller=controller,
    )
    controller_before = dict(vars(controller))

    with pytest.raises(RuntimeError, match="controller failed"):
        runtime.advance(TimeWindowRequest(
            request_id=1,
            generation=0,
            t_start=0.0,
            t_end=0.2,
            dt_initial=0.1,
        ))

    assert state.time == 0.0
    assert runtime.current_time == 0.0
    assert vars(controller) == controller_before


def test_managed_runtime_commit_gate_rolls_back_completed_window():
    state = ManagedState()

    def advance(value, dt):
        value.time += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0)

    controller = STController(dt_min=1.0e-9)
    runtime = ManagedTimeRuntime(
        state,
        advance,
        dx=1.0,
        targets=Targets(1.0, 1.0, 1.0),
        controller=controller,
    )
    controller_before = dict(vars(controller))
    request = TimeWindowRequest(5, 0, 0.0, 0.2, 0.1)
    observed = []

    with pytest.raises(RuntimeError, match="commit gate rejected"):
        runtime.advance(
            request,
            commit_gate=lambda gated_request, result: (
                observed.append((gated_request, result.advanced)) or False
            ),
        )

    assert observed == [(request, pytest.approx(0.2))]
    assert state.time == 0.0
    assert runtime.current_time == 0.0
    assert runtime.last_request_id is None
    assert runtime.last_report is None
    assert vars(controller) == controller_before

    report = runtime.advance(request, commit_gate=lambda _request, _result: True)
    assert report.exact_landing
    assert state.time == pytest.approx(0.2)
    assert runtime.current_time == pytest.approx(0.2)


def test_managed_runtime_restores_controller_even_if_state_rollback_fails():
    class FailedRestoreState(ManagedState):
        def restore(self, snapshot) -> None:
            raise OSError("native checkpoint restore failed")

    state = FailedRestoreState()
    controller = STController(dt_min=1.0e-9)
    runtime = ManagedTimeRuntime(
        state,
        lambda value, dt: (
            setattr(value, "time", value.time + float(dt))
            or (True, Metrics(0.0, 0.0, 0.0, 0.0))
        ),
        dx=1.0,
        targets=Targets(1.0, 1.0, 1.0),
        controller=controller,
    )
    controller_before = dict(vars(controller))

    with pytest.raises(RuntimeError, match="state rollback failed") as failure:
        runtime.advance(
            TimeWindowRequest(1, 0, 0.0, 0.1, 0.1),
            commit_gate=lambda _request, _result: False,
        )

    assert isinstance(failure.value.__cause__, OSError)
    assert vars(controller) == controller_before
    assert runtime.current_time == 0.0
    assert runtime.last_request_id is None


def test_managed_runtime_rejects_stale_or_discontinuous_requests():
    state = ManagedState()
    runtime = ManagedTimeRuntime(
        state,
        lambda value, dt: (
            True,
            Metrics(0.0, 0.0, 0.0, 0.0),
        ),
        dx=1.0,
        targets=Targets(1.0, 1.0, 1.0),
        generation=3,
        initial_time=2.0,
    )

    with pytest.raises(RuntimeError, match="stale"):
        runtime.advance(TimeWindowRequest(1, 2, 2.0, 2.1, 0.1))
    with pytest.raises(RuntimeError, match="current committed time"):
        runtime.advance(TimeWindowRequest(1, 3, 2.1, 2.2, 0.1))

    runtime.supersede(4, at_time=5.0)
    assert runtime.generation == 4
    assert math.isclose(runtime.current_time, 5.0)


def test_managed_runtime_rejects_committed_request_replay():
    state = ManagedState()

    def advance(value, dt):
        value.time += float(dt)
        return True, Metrics(0.0, 0.0, 0.0, 0.0)

    runtime = ManagedTimeRuntime(
        state,
        advance,
        dx=1.0,
        targets=Targets(1.0, 1.0, 1.0),
    )
    runtime.advance(TimeWindowRequest(7, 0, 0.0, 0.1, 0.1))

    with pytest.raises(RuntimeError, match="replayed or reordered"):
        runtime.advance(TimeWindowRequest(7, 0, 0.1, 0.2, 0.1))

    assert runtime.current_time == pytest.approx(0.1)
    assert state.time == pytest.approx(0.1)


@pytest.mark.parametrize(
    "window, message",
    [
        (TimeWindowRequest(0, 0, 1.0, 1.0, 0.1), "positive window"),
        (TimeWindowRequest(0, 0, 0.0, 1.0, 0.0), "initial dt"),
        (
            TimeWindowRequest(0, 0, 0.0, 1.0, 0.1, (0.7, 0.3)),
            "unique and ordered",
        ),
        (
            TimeWindowRequest(0, 0, 0.0, 1.0, 0.1, (1.0,)),
            "inside the window",
        ),
    ],
)
def test_time_window_request_validation(window, message):
    with pytest.raises(ValueError, match=message):
        window.validate()
