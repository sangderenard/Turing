from __future__ import annotations

import math

import pytest

from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner
from src.common.dt_system.engine_api import EngineRegistration
from src.common.dt_system.dt_solver import (
    BisectSolverConfig,
    solve_window_bisect,
)
from src.common.dt_system.state_table import StateTable


class TransactionalEngine:
    def __init__(self) -> None:
        self.position = 0.0
        self.world_time = 0.0
        self.observer_time = 0.0
        self.attempted: list[float] = []

    def snapshot(self):
        return self.position

    def restore(self, snapshot) -> None:
        self.position = float(snapshot)

    def sync_from_state(self, table: StateTable) -> None:
        value = table.get("engine", "probe", "position")
        if value is not None:
            self.position = float(value)

    def publish_to_state(self, table: StateTable) -> None:
        table.set("engine", "probe", "position", self.position)

    def step(self, dt: float, state=None, state_table=None):
        self.attempted.append(float(dt))
        self.position += float(dt)
        self.world_time += float(dt)
        self.observer_time += float(dt)
        metrics = Metrics(
            max_vel=1.0,
            max_flux=0.0,
            div_inf=float(dt),
            mass_err=0.0,
        )
        return True, metrics, state


def test_state_table_snapshot_restore_isolated_and_in_place():
    table = StateTable()
    table.set("scope", "name", "value", {"items": [1, 2]})
    identity = table.register_identity((1.0, 2.0), mass=3.0)
    snapshot = table.snapshot()

    table.get("scope", "name", "value")["items"].append(9)
    table.update_identity(identity, pos=(8.0, 9.0))
    table.restore(snapshot)

    assert table.get("scope", "name", "value") == {"items": [1, 2]}
    assert table.get_identity(identity)["pos"] == (1.0, 2.0)


def test_bisection_candidates_rollback_and_commits_land_exactly():
    engine = TransactionalEngine()
    table = StateTable()
    table.set("engine", "probe", "position", 0.0)
    config = BisectSolverConfig(
        target=0.1,
        eps=1.0e-10,
        dt_min=0.01,
        dt_max=0.2,
        field="div_inf",
        monotonic="increase",
    )

    metrics = solve_window_bisect(
        engine,
        0.25,
        config,
        state_table=table,
        registration_name="probe",
    )

    assert len(engine.attempted) > 3
    assert math.isclose(engine.position, 0.25, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(engine.world_time, 0.25, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(engine.observer_time, 0.25, rel_tol=0.0, abs_tol=1.0e-12)
    assert table.get("engine", "probe", "position") == pytest.approx(0.25)
    assert metrics.div_inf == pytest.approx(0.05)


def test_bisection_requires_transactional_engine_by_default():
    class UnsafeEngine:
        def step(self, dt):
            return True, Metrics(0.0, 0.0, dt, 0.0)

    with pytest.raises(ValueError, match="snapshot/restore"):
        solve_window_bisect(
            UnsafeEngine(),
            0.1,
            BisectSolverConfig(target=0.05),
            state_table=StateTable(),
        )


@pytest.mark.parametrize(
    "config, message",
    [
        (BisectSolverConfig(target=1.0, dt_min=0.0), "dt_min"),
        (BisectSolverConfig(target=1.0, dt_max=0.0), "dt_max"),
        (BisectSolverConfig(target=1.0, monotonic="sideways"), "monotonic"),
        (
            BisectSolverConfig(target=1.0, require_snapshot=False),
            "transactional candidate",
        ),
    ],
)
def test_bisection_rejects_invalid_policy(config, message):
    with pytest.raises(ValueError, match=message):
        solve_window_bisect(
            TransactionalEngine(),
            0.1,
            config,
            state_table=StateTable(),
        )


def test_graphbuilder_bisection_path_uses_explicit_state_table():
    engine = TransactionalEngine()
    table = StateTable()
    table.set("engine", "probe", "position", 0.0)
    targets = Targets(1.0, 1.0, 1.0)
    registration = EngineRegistration(
        name="probe",
        engine=engine,
        targets=targets,
        dx=1.0,
        localize=False,
        solver_config=BisectSolverConfig(
            target=0.1,
            eps=1.0e-10,
            dt_min=0.01,
            dt_max=0.2,
            field="div_inf",
        ),
    )
    root = GraphBuilder(
        STController(dt_min=1.0e-9), targets, 1.0
    ).round(0.25, [registration], state_table=table)

    result = MetaLoopRunner(state_table=table).run_round(
        root, state_table=table
    )

    assert result.advanced == pytest.approx(0.25)
    assert engine.position == pytest.approx(0.25)
    assert table.get("engine", "probe", "position") == pytest.approx(0.25)
