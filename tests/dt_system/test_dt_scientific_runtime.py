from __future__ import annotations

from src.common.tensors import AbstractTensor

import math

import pytest

from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import (
    AdvanceNode,
    ControllerNode,
    MetaLoopRunner,
    RoundNode,
    StateNode,
)
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.state_table import StateTable


class ClockState:
    def __init__(self) -> None:
        self.time = 0.0


def test_scientific_graph_rolls_back_named_error_and_lands_on_event():
    state_node = StateNode(ClockState(), label="wave-state")
    table = StateTable()

    def advance(state, dt, *, realtime=False, state_table=None):
        state.time += float(dt)
        accepted_count = state_table.get("test", "advance", "count") or 0
        state_table.set("test", "advance", "count", accepted_count + 1)
        return (
            True,
            Metrics(
                max_vel=1.0,
                max_flux=0.0,
                div_inf=0.0,
                mass_err=0.0,
                error_channels=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(dt), 0.0]),
                error_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            ),
            state,
        )

    leaf = AdvanceNode(advance, state_node, label="wave-advance")
    plan = SuperstepPlan(
        round_max=0.2,
        dt_init=0.2,
        event_boundaries=(0.15,),
    )
    controller = ControllerNode(
        STController(dt_min=1.0e-8),
        Targets(
            cfl=1.0,
            div_max=1.0,
            mass_max=1.0,
            error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
        dx=1.0,
    )
    root = RoundNode(
        plan,
        controller,
        [leaf],
        label="camera-slice",
        state_table=table,
    )

    result = MetaLoopRunner(state_table=table).run_round(
        root, state_table=table
    )

    assert math.isclose(result.advanced, 0.2, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(
        state_node.state.time, 0.2, rel_tol=0.0, abs_tol=1.0e-15
    )
    assert result.rejected_attempts >= 1
    assert result.attempted_dts[0] == 0.15
    assert result.landed_boundaries == (0.15,)
    # The rejected candidate increment was rolled back with the StateTable.
    assert table.get("test", "advance", "count") == len(result.accepted_dts)


def test_parallel_children_conservatively_combine_scientific_errors():
    state = StateNode(object())

    def advance_with(name, error):
        def advance(value, dt, *, realtime=False, state_table=None):
            return True, Metrics(
                0.0,
                0.0,
                0.0,
                0.0,
                error_channels=AbstractTensor.tensor([0.0] * 15 + [error if name == "phase" else 0.0, 0.0, error if name == "power" else 0.0]),
                error_present=AbstractTensor.tensor([0.0] * 15 + [float(name == "phase"), 0.0, float(name == "power")]),
            ), value

        return advance

    root = RoundNode(
        SuperstepPlan(0.1, 0.1),
        ControllerNode(
            STController(dt_min=1.0e-8),
            Targets(
                cfl=1.0,
                div_max=1.0,
                mass_max=1.0,
                error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
                error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
            ),
            1.0,
        ),
        [
            AdvanceNode(advance_with("phase", 0.2), state, "phase"),
            AdvanceNode(advance_with("power", 0.3), state, "power"),
        ],
        schedule="parallel",
        label="coupled",
        state_table=StateTable(),
    )

    result = MetaLoopRunner(state_table=root.state_table).run_round(root)

    assert result.metrics.error_channels.tolist()[15:] == [0.2, 0.0, 0.3]
    assert result.metrics.error_present.tolist()[15:] == [1.0, 0.0, 1.0]


def test_scientific_graph_rolls_back_entire_failed_window():
    state_node = StateNode(ClockState(), label="state")
    table = StateTable()

    def advance(state, dt, *, realtime=False, state_table=None):
        state.time += float(dt)
        return (
            True,
            Metrics(
                0.0,
                0.0,
                0.0,
                0.0,
                hard_failure=state.time > 0.1,
            ),
            state,
        )

    root = RoundNode(
        SuperstepPlan(0.2, 0.1),
        ControllerNode(
            STController(dt_min=1.0e-12),
            Targets(1.0, 1.0, 1.0),
            1.0,
        ),
        [AdvanceNode(advance, state_node)],
        label="atomic-window",
        state_table=table,
    )
    runner = MetaLoopRunner(state_table=table)

    with pytest.raises(RuntimeError, match="controller failed"):
        runner.run_round(root)

    assert state_node.state.time == 0.0
    assert runner.get_attempted_dts(root) == []
