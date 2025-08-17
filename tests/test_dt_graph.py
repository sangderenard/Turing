import math
import pytest

import math
import pytest

from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_graph import (
    StateNode,
    AdvanceNode,
    ControllerNode,
    RoundNode,
    MetaLoopRunner,
)
from src.cells.bath.dt_controller import STController, Targets
from src.common.dt_system.dt_scaler import Metrics


class DummyState:
    def __init__(self):
        self.t = 0.0


def make_advance(max_vel: float):
    def advance(state: DummyState, dt: float, *, realtime: bool = False):
        state.t += float(dt)
        m = Metrics(max_vel=max_vel, max_flux=max_vel, div_inf=0.0, mass_err=0.0)
        return True, m, state
    return advance


@pytest.mark.dt
@pytest.mark.dt_graph
@pytest.mark.fast
def test_round_with_single_advance_node():
    state = DummyState()
    s_node = StateNode(state=state)
    a_node = AdvanceNode(advance=make_advance(max_vel=2.0), state=s_node)

    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)
    plan = SuperstepPlan(round_max=0.2, dt_init=0.05)
    round_root = RoundNode(plan=plan, controller=ctrl, children=[a_node])

    runner = MetaLoopRunner()
    res = runner.run_round(round_root)

    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps)
    assert isinstance(res.dt_next, float)
    assert state.t > 0.0


@pytest.mark.dt
@pytest.mark.dt_graph
def test_nested_rounds_delegate_dt():
    outer = DummyState()
    inner = DummyState()

    s_inner = StateNode(state=inner)
    a_inner = AdvanceNode(advance=make_advance(max_vel=10.0), state=s_inner)

    inner_ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=0.1)
    inner_round = RoundNode(plan=SuperstepPlan(round_max=0.0, dt_init=0.01), controller=inner_ctrl, children=[a_inner])

    s_outer = StateNode(state=outer)
    # Outer advance just updates metrics moderately; real sims would couple state changes too
    a_outer = AdvanceNode(advance=make_advance(max_vel=1.0), state=s_outer)

    outer_ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)
    plan_outer = SuperstepPlan(round_max=0.2, dt_init=0.2)

    # Outer round delegates its dt to inner round first, then performs a simple outer advance
    outer_round = RoundNode(plan=plan_outer, controller=outer_ctrl, children=[inner_round, a_outer])

    runner = MetaLoopRunner()
    res = runner.run_round(outer_round)

    assert math.isclose(res.advanced, plan_outer.round_max, rel_tol=0, abs_tol=plan_outer.eps)
    # Inner advanced fully
    assert math.isclose(inner.t, plan_outer.round_max, rel_tol=0, abs_tol=plan_outer.eps)
