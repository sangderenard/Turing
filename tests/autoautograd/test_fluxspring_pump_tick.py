import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick


def _make_spec() -> FluxSpringSpec:
    nodes = [
        NodeSpec(
            id=0,
            p0=AT.get_tensor([0.0]),
            v0=AT.get_tensor([0.0]),
            mass=AT.tensor(1.0),
            ctrl=NodeCtrl(),
            scripted_axes=[0],
        ),
        NodeSpec(
            id=1,
            p0=AT.get_tensor([0.0]),
            v0=AT.get_tensor([0.0]),
            mass=AT.tensor(1.0),
            ctrl=NodeCtrl(),
            scripted_axes=[0],
        ),
    ]
    edge = EdgeSpec(
        src=0,
        dst=1,
        transport=EdgeTransport(
            kappa=AT.tensor(1.0),
            k=AT.tensor(1.0),
            l0=AT.tensor(1.0),
            lambda_s=AT.tensor(1.0),
            x=AT.tensor(0.0),
        ),
        ctrl=EdgeCtrl(),
    )
    dec = DECSpec(D0=[[-1.0, 1.0]], D1=[])
    return FluxSpringSpec(
        version="test",
        D=1,
        nodes=nodes,
        edges=[edge],
        faces=[],
        dec=dec,
        gamma=AT.tensor(0.0),
    )


def test_pump_tick_injection_leak0():
    spec = _make_spec()
    psi = AT.zeros(2)
    psi_next, _ = pump_tick(psi, spec, eta=0.0, external={0: AT.tensor(1.0)}, leak=0.0)
    assert psi_next.shape[0] == 2
    assert float(AT.get_tensor(psi_next)[0]) == pytest.approx(1.0)


def test_pump_tick_leak_decay():
    spec = _make_spec()
    psi = AT.get_tensor([1.0, -1.0])
    psi_next, _ = pump_tick(psi, spec, eta=0.0, leak=0.2)
    vals = AT.get_tensor(psi_next)
    assert float(vals[0]) == pytest.approx(0.8)
    assert float(vals[1]) == pytest.approx(-0.8)


def test_pump_tick_saturate():
    spec = _make_spec()
    psi = AT.get_tensor([2.0, -2.0])
    sat = lambda x: AT.clip(x, -1.0, 1.0)
    psi_next, _ = pump_tick(psi, spec, eta=0.0, leak=0.0, saturate=sat)
    vals = AT.get_tensor(psi_next)
    assert float(vals[0]) == pytest.approx(1.0)
    assert float(vals[1]) == pytest.approx(-1.0)


def test_pump_tick_lorentz():
    spec = _make_spec()
    psi = AT.get_tensor([0.5, 0.0])
    eta = 0.1
    c = 2.0
    psi_next, stats = pump_tick(psi, spec, eta=eta, lorentz_c=c)
    delta = stats["delta"]
    expected = psi + eta * (delta / AT.sqrt(1.0 - (psi / c) ** 2))
    vals = AT.get_tensor(psi_next)
    exp_vals = AT.get_tensor(expected)
    assert float(vals[0]) == pytest.approx(float(exp_vals[0]))
    assert float(vals[1]) == pytest.approx(float(exp_vals[1]))


def test_pump_tick_norm_node():
    spec = _make_spec()
    psi = AT.get_tensor([0.5, -0.5])
    eta = 1.0
    _, stats_off = pump_tick(psi, spec, eta=eta)
    _, stats_norm = pump_tick(psi, spec, eta=eta, norm="node")
    delta_off = AT.get_tensor(stats_off["delta"])
    delta_norm = AT.get_tensor(stats_norm["delta"])
    q_off = AT.get_tensor(stats_off["q"])
    q_norm = AT.get_tensor(stats_norm["q"])
    assert float(delta_norm[0]) == pytest.approx(float(delta_off[0]) / 2)
    assert float(delta_norm[1]) == pytest.approx(float(delta_off[1]) / 2)
    assert float(q_norm[0]) == pytest.approx(float(q_off[0]) / 2)
