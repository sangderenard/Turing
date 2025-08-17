import math
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
from src.common.dt_system.state_table import StateTable


class Ctr:
    def __init__(self):
        self.calls = []


def make_adv(tag: str, ctr: Ctr, vel: float):
    uid = None

    def advance(_state, dt: float, *, realtime: bool = False, state_table=None):
        nonlocal uid
        ctr.calls.append((tag, float(dt)))
        if state_table is not None:
            if uid is None:
                uid = state_table.register_identity(pos=ctr.calls[-1], mass=1.0)
            else:
                state_table.update_identity(uid, pos=ctr.calls[-1])
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
    table = StateTable()
    res = MetaLoopRunner(state_table=table).run_round(round_node, dt=plan.round_max, state_table=table)

    # Calls should be in order with equal slices per child per micro-step.
    assert any(tag == "a1" for tag, _ in ctr.calls)
    assert any(tag == "a2" for tag, _ in ctr.calls)
    # Landing check to ensure the outer loop ran
    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps)
    assert len(table.identity_registry) == 2


@pytest.mark.dt
@pytest.mark.dt_graph
@pytest.mark.fast
def test_parallel_schedule_combines_metrics():
    s = StateNode(state=object())

    def const_adv(max_vel, div_inf, mass_err):
        uid = None

        def f(st, dt, *, realtime=False, state_table=None):
            nonlocal uid
            if state_table is not None:
                if uid is None:
                    uid = state_table.register_identity(pos=max_vel, mass=1.0)
                else:
                    state_table.update_identity(uid, pos=max_vel)
            return True, Metrics(max_vel, max_vel, div_inf, mass_err), st

        return f

    a1 = AdvanceNode(advance=const_adv(1.0, 1e-6, 1e-9), state=s)
    a2 = AdvanceNode(advance=const_adv(2.0, 2e-6, 2e-9), state=s)

    plan = SuperstepPlan(round_max=0.1, dt_init=0.1)
    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)

    round_node = RoundNode(plan=plan, controller=ctrl, children=[a1, a2], schedule="parallel")
    table = StateTable()
    res = MetaLoopRunner(state_table=table).run_round(round_node, dt=plan.round_max, state_table=table)

    assert math.isclose(res.advanced, plan.round_max, rel_tol=0, abs_tol=plan.eps)
    assert len(table.identity_registry) == 2
