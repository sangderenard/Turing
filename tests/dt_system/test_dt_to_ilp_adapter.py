import pytest

import pytest

from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_graph import (
    StateNode,
    AdvanceNode,
    ControllerNode,
    RoundNode,
)
from src.common.dt_system.dt_process_adapter import schedule_dt_round
from src.cells.bath.dt_controller import STController, Targets
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.state_table import StateTable


class S:
    def __init__(self):
        self.t = 0.0


def adv(max_vel: float):
    def f(_state: S, dt: float, *, state_table=None, realtime: bool = False):
        if state_table is not None:
            state_table.set("adv", str(id(_state)), "t", getattr(_state, "t", 0.0))
        return True, Metrics(max_vel=max_vel, max_flux=max_vel, div_inf=0.0, mass_err=0.0), _state

    return f


@pytest.mark.dt_graph
@pytest.mark.fast
def test_adapter_builds_levels_and_interference_sequential():
    s = StateNode(state=S())
    a1 = AdvanceNode(advance=adv(1.0), state=s, label="a1")
    a2 = AdvanceNode(advance=adv(2.0), state=s, label="a2")

    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)
    plan = SuperstepPlan(round_max=0.1, dt_init=0.1)
    table = StateTable()
    root = RoundNode(plan=plan, controller=ctrl, children=[a1, a2], schedule="sequential", state_table=table)

    levels, ig, lifespans, G = schedule_dt_round(root, method="asap", order="dependency")

    # sequential should create an edge a1->a2 and levels ordered
    n1, n2 = id(a1), id(a2)
    assert G.has_edge(n1, n2)
    assert levels[n1] <= levels[n2]
    # lifespans overlap due to edge, so interference for union schedule
    assert ig.has_edge(n1, n2)


@pytest.mark.dt_graph
@pytest.mark.fast
def test_adapter_parallel_no_dependency_edge():
    s = StateNode(state=S())
    a1 = AdvanceNode(advance=adv(1.0), state=s, label="a1")
    a2 = AdvanceNode(advance=adv(2.0), state=s, label="a2")

    ctrl = ControllerNode(ctrl=STController(dt_min=1e-6), targets=Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6), dx=1.0)
    plan = SuperstepPlan(round_max=0.1, dt_init=0.1)
    table = StateTable()
    root = RoundNode(plan=plan, controller=ctrl, children=[a1, a2], schedule="parallel", state_table=table)

    levels, ig, lifespans, G = schedule_dt_round(root, method="asap", order="dependency")

    n1, n2 = id(a1), id(a2)
    assert not G.has_edge(n1, n2)
    assert not G.has_edge(n2, n1)
    # Depending on ASAP, both can be at same level; interference graph may still connect
    # due to lifespan union (start-1 .. level). We just assert keys exist.
    assert n1 in lifespans and n2 in lifespans
