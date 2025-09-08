import math
import threading
import time

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd import spring_async_toy as toy
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_graph import (
    AdvanceNode,
    ControllerNode,
    MetaLoopRunner,
    RoundNode,
    StateNode,
)
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.roundnode_engine import RoundNodeEngine
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.threaded_system import ThreadedSystemEngine
from src.common.dt_system.spectral_dampener import spectral_inertia


def _build_simple_system():
    AT = AbstractTensor
    n0 = toy.Node(
        id=0,
        p=AT.tensor([0.0, 0.0, 0.0]),
        v=AT.zeros(3, float),
        geom_mask=AT.zeros(3, float),
    )
    n1 = toy.Node(
        id=1,
        p=AT.tensor([0.0, 1.5, 0.0]),
        v=AT.zeros(3, float),
        geom_mask=AT.tensor([0.0, 1.0, 0.0]),
    )
    for n in (n0, n1):
        n.commit()
        n.hist_p.append(n.p.copy())
    edge = toy.Edge(
        key=(0, 1, "spring"),
        i=0,
        j=1,
        op_id="spring",
        l0=AT.tensor(0.5),
        k=AT.tensor(2.0),
    )
    sys = toy.SpringRepulsorSystem([n0, n1], [edge], eta=0.0, gamma=1.0, dt=0.01)
    # Ensure initial positions persist after system construction
    sys.nodes[1].p = AT.tensor([0.0, 1.5, 0.0])
    return sys


def _build_round(sys: toy.SpringRepulsorSystem, table: StateTable) -> RoundNode:
    engine = toy.SpringDtEngine(sys)

    def _adv(state, dt, *, realtime=False, state_table=None):
        return engine.step(dt, state_table=state_table)

    state = StateNode(state=None)
    adv = AdvanceNode(advance=_adv, state=state)
    ctrl = ControllerNode(
        ctrl=STController(dt_min=1e-6),
        targets=Targets(cfl=1.0, div_max=1.0, mass_max=1.0),
        dx=1.0,
    )
    plan = SuperstepPlan(round_max=0.01, dt_init=0.01)
    return RoundNode(plan=plan, controller=ctrl, children=[adv], state_table=table)


def test_meta_loop_runner_moves_free_node():
    sys = _build_simple_system()
    table = StateTable()
    round_node = _build_round(sys, table)
    runner = MetaLoopRunner(state_table=table)
    x0 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[0])
    y0 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[1])
    z0 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[2])
    for _ in range(5):
        runner.run_round(round_node, dt=0.01, state_table=table)
    x1 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[0])
    y1 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[1])
    z1 = float(AbstractTensor.get_tensor(sys.nodes[1].p)[2])
    assert not math.isclose(y0, y1)
    assert math.isclose(x0, x1, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(z0, z1, rel_tol=1e-6, abs_tol=1e-6)


def test_spectral_inertia_reduces_velocity_norm():
    AT = AbstractTensor
    dt = 0.1
    hist = [AT.tensor([math.sin(dt * i), math.cos(dt * i)]) for i in range(128)]
    resp, _, _ = spectral_inertia(hist, dt)
    v_last = hist[-1] - hist[-2]
    energy_no = float(AT.get_tensor((v_last * v_last).sum()).item())
    diff = v_last - resp * 1e-4
    energy_damped = float(AT.get_tensor((diff * diff).sum()).item())
    assert energy_damped < energy_no


def test_threaded_engine_steps_independently():
    sys = _build_simple_system()
    table = StateTable()
    round_node = _build_round(sys, table)
    rne = RoundNodeEngine(inner=round_node, runner=MetaLoopRunner(state_table=table))

    def capture():
        return {
            "pos": [
                AbstractTensor.get_tensor(n.p).tolist() for n in sys.nodes.values()
            ]
        }

    eng = ThreadedSystemEngine(rne, capture=capture, max_queue=2)
    try:
        def drive():
            for _ in range(3):
                eng.step(0.01, state_table=table)
                time.sleep(0.01)

        t = threading.Thread(target=drive)
        t.start()
        t.join(timeout=1.0)
        time.sleep(0.05)
        assert not eng.output_queue.empty()
        frame = eng.output_queue.get_nowait()
        assert isinstance(frame["pos"], list)
        assert len(frame["pos"]) == len(sys.nodes)
    finally:
        eng.stop()
