import math
import threading
import time

import pytest

pytestmark = pytest.mark.xfail(
    reason="spring_async_toy transitioning to FluxSpring wrappers",
    strict=False,
)

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
    x0 = float(sys.nodes[1].p[0])
    y0 = float(sys.nodes[1].p[1])
    z0 = float(sys.nodes[1].p[2])
    for _ in range(5):
        runner.run_round(round_node, dt=0.01, state_table=table)
    x1 = float(sys.nodes[1].p[0])
    y1 = float(sys.nodes[1].p[1])
    z1 = float(sys.nodes[1].p[2])
    assert not math.isclose(y0, y1)
    assert math.isclose(x0, x1, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(z0, z1, rel_tol=1e-6, abs_tol=1e-6)


def test_boundary_targets_clamp_x_and_z():
    AT = AbstractTensor
    n_in = toy.Node(
        id=0,
        p=AT.tensor([1.0, 2.0, 0.0]),
        v=AT.zeros(3, float),
        geom_mask=AT.ones(3, float),
    )
    n_out = toy.Node(
        id=1,
        p=AT.tensor([0.0, -1.0, 2.0]),
        v=AT.zeros(3, float),
        geom_mask=AT.ones(3, float),
    )
    for n in (n_in, n_out):
        n.commit()
        n.hist_p.append(n.p.copy())
    sys = toy.SpringRepulsorSystem([n_in, n_out], [], eta=0.0, gamma=0.0, dt=0.01)
    sys.add_boundary(
        toy.BoundaryPort(
            nid=0,
            alpha=10.0,
            target_fn=lambda t: AT.tensor([1.0, 0.0, 0.0]),
            axis_mask=AT.tensor([1.0, 0.0, 0.0]),
        )
    )
    sys.add_boundary(
        toy.BoundaryPort(
            nid=1,
            alpha=10.0,
            target_fn=lambda t: AT.tensor([0.0, 0.0, 2.0]),
            axis_mask=AT.tensor([0.0, 0.0, 1.0]),
        )
    )
    table = StateTable()
    round_node = _build_round(sys, table)
    runner = MetaLoopRunner(state_table=table)
    y0_in = float(sys.nodes[0].p[1])
    y0_out = float(sys.nodes[1].p[1])
    for _ in range(5):
        runner.run_round(round_node, dt=0.01, state_table=table)
    p_in = sys.nodes[0].p
    p_out = sys.nodes[1].p
    assert math.isclose(float(p_in[0]), 1.0, rel_tol=1e-6, abs_tol=1e-3)
    assert math.isclose(float(p_out[2]), 2.0, rel_tol=1e-6, abs_tol=1e-3)
    assert math.isclose(float(p_in[1]), y0_in, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(p_out[1]), y0_out, rel_tol=1e-6, abs_tol=1e-6)


def test_spectral_inertia_reduces_velocity_norm():
    AT = AbstractTensor
    dt = 0.1
    hist = [AT.tensor([math.sin(dt * i), math.cos(dt * i)]) for i in range(128)]
    resp, _, _ = spectral_inertia(hist, dt)
    v_last = hist[-1] - hist[-2]
    energy_no = float((v_last * v_last).sum())
    diff = v_last - resp * 1e-4
    energy_damped = float((diff * diff).sum())
    assert energy_damped < energy_no


def test_spectral_inertia_passthrough_with_short_history():
    AT = AbstractTensor
    dt = 0.1
    # Fewer samples than the minimum FFT window should yield a zero response
    hist = [AT.tensor([0.0, 0.0]) for _ in range(10)]
    resp, J, bands = spectral_inertia(hist, dt)
    assert resp.abs().sum().item() == 0.0
    assert J.abs().sum().item() == 0.0
    assert bands == []


def test_threaded_engine_steps_independently():
    sys = _build_simple_system()
    table = StateTable()
    round_node = _build_round(sys, table)
    rne = RoundNodeEngine(inner=round_node, runner=MetaLoopRunner(state_table=table))

    def capture():
        return {
            "pos": [
                n.p.tolist() for n in sys.nodes.values()
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
