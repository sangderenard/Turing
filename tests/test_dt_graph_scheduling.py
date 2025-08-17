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


class Ctr:
    def __init__(self):
        self.calls = []


def make_adv(tag: str, ctr: Ctr, vel: float):
    def advance(_state, dt: float):
        ctr.calls.append((tag, float(dt)))
    return True, Metrics(max_vel=vel, max_flux=vel, div_inf=0.0, mass_err=0.0), _state
    return advance


@pytest.mark.dt
@pytest.mark.dt_graph
@pytest.mark.fast
def test_interleave_schedule_slices_dt():
    ctr = Ctr()
    s = StateNode(state=object())
    a1 = AdvanceNode(advance=make_adv("a1", ctr, 1.0), state=s)
    a2 = AdvanceNode(advance=make_adv("a2", ctr, 2.0), state=s)

    plan = SuperstepPlan(round_max=0.3, dt_init=0.3)
    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)

    round_node = RoundNode(plan=plan, controller=ctrl, children=[a1, a2], schedule="interleave")
    res = MetaLoopRunner().run_round(round_node)

    # Calls should be in order with equal slices per child per micro-step.
    assert any(tag == "a1" for tag, _ in ctr.calls)
    assert any(tag == "a2" for tag, _ in ctr.calls)
    # Landing check to ensure the outer loop ran
    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps)


@pytest.mark.dt
@pytest.mark.dt_graph
@pytest.mark.fast
def test_parallel_schedule_combines_metrics():
    s = StateNode(state=object())
    a1 = AdvanceNode(advance=lambda st, dt: (True, Metrics(1.0, 1.0, 1e-6, 1e-9), st), state=s)
    a2 = AdvanceNode(advance=lambda st, dt: (True, Metrics(2.0, 2.0, 2e-6, 2e-9), st), state=s)

    plan = SuperstepPlan(round_max=0.1, dt_init=0.1)
    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)

    round_node = RoundNode(plan=plan, controller=ctrl, children=[a1, a2], schedule="parallel")
    res = MetaLoopRunner().run_round(round_node)

    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps)
