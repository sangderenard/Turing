"""Error headroom and rejected trials steer the existing PI and rollback."""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from llvm_dt_system import advance_round, instantiate_system, piece_leaf
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets, _propose_dt_pen, step_with_dt_control_used
from src.common.dt_system.dt_graph import ControllerNode, RoundNode
from src.common.dt_system.dt_scaler import Metrics, _scalar
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields
from src.compiler.native_package import piece_from_law
from src.compiler.symbolic_equation_compiler import compile_sympy_equations


ERROR = "feedback_trial_error"
CHANNEL_NAMES = (*DT_CHANNEL_NAMES, ERROR)


@pytest.mark.parametrize("power", (6, 8))
def test_native_error_feedback_recovers_headroom_and_handles_command_change(tmp_path, power, capsys):
    x, dt, gain = sp.symbols("x dt gain")
    law = compile_sympy_equations((
        sp.Eq(sp.Symbol("x_next"), x + dt, evaluate=False),
        sp.Eq(sp.Symbol(ERROR), (gain * dt) ** power, evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), 1, evaluate=False),
    ), name=f"error_feedback_{power}")
    piece = piece_from_law(law, f"error_feedback_{power}", 1,
                           directory=tmp_path).for_runtime()
    targets = Targets(1000.0, 1.0, 1.0, **channel_fields(
        {ERROR: 1.0}, names=CHANNEL_NAMES, limits=True))
    root = RoundNode(SuperstepPlan(0.5, 0.05, rollback=True),
                     ControllerNode(STController(), targets, 1000.0),
                     children=[piece_leaf(piece)])
    state = instantiate_system(root, {"x": np.zeros(1), "gain": np.array([8.0])},
                               channel_names=CHANNEL_NAMES)
    owned_x = state.x
    native_advance = state.program["advance_pieces"]
    attempts = []

    def observe(actual_state, step):
        before = float(actual_state.x[0])
        ok, metrics = native_advance(actual_state, step)
        attempts.append((float(step), before, float(metrics.error_channels[-1])))
        return ok, metrics

    state.program["advance_pieces"] = observe
    for phase, command in enumerate((8.0, 32.0, 0.0), 1):
        state.gain[...] = command
        attempts.clear()
        advanced, continuation, _ = advance_round(state)
        committed = (phase - 1) * 0.5
        for step, before, ratio in attempts:
            assert before == pytest.approx(committed, abs=2e-15)
            if ratio <= 1.0:
                committed += step
        assert advanced == 0.5
        assert state.x is owned_x
        assert float(state.x[0]) == pytest.approx(phase * 0.5, abs=2e-15)
        rejected = sum(ratio > 1.0 for _, _, ratio in attempts)
        with capsys.disabled():
            print(f"power={power} gain={command} attempts={len(attempts)} "
                  f"rejected={rejected} acc={_scalar(state.controller.acc):.12g} "
                  f"dt_range=({min(row[0] for row in attempts):.12g},"
                  f"{max(row[0] for row in attempts):.12g}) "
                  f"continuation={continuation:.12g}", flush=True)
        assert len(attempts) < 200
        if phase == 1:
            assert max(step for step, _, ratio in attempts if ratio <= 1) > 0.05
        elif phase == 2:
            assert rejected > 0
        else:
            assert rejected == 0 and len(attempts) < 5
            assert continuation > 0.5


def test_failure_without_error_estimate_still_refines_with_unit_oscillation_shrink():
    class State:
        value = 0.0

        def copy_shallow(self):
            return self.value

        def restore(self, saved):
            self.value = saved

    state = State()
    attempts = []

    def fail(actual_state, dt):
        attempts.append(float(dt))
        actual_state.value += float(dt)
        return False, Metrics(0.0, 0.0, 0.0, 0.0)

    metrics, _, used = step_with_dt_control_used(
        state, 0.1, 1.0, Targets(1.0, 1.0, 1.0), STController(shrink=1.0),
        fail, max_retries=None, rollback=True, allow_unresolved=False,
    )
    assert metrics.hard_failure and used == 0.0
    assert state.value == 0.0
    assert 1 < len(attempts) < 100
    assert all(right < left for left, right in zip(attempts, attempts[1:]))


def test_declared_distribution_keeps_its_three_argument_contract():
    metrics = Metrics(2.0, 0.0, 0.0, 0.0)
    targets = Targets(1.0, 1.0, 1.0)
    calls = []

    def distribution(observed, configured, length):
        calls.append((observed, configured, length))
        return 0.037

    assert _propose_dt_pen(metrics, targets, 4.0, distribution, 0.5) == 0.037
    assert calls == [(metrics, targets, 4.0)]
