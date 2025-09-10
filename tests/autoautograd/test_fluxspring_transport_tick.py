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
from src.common.tensors.autoautograd.fluxspring.fs_dec import transport_tick


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


def test_transport_tick_shapes():
    spec = _make_spec()
    psi = AT.zeros(2)
    P = AT.get_tensor([[0.0], [1.0]])
    psi_next, stats = transport_tick(psi, spec, eta=0.1, P=P)
    assert psi_next.ndim == 1
    for key in ["q", "dpsi", "G", "R"]:
        assert stats[key].ndim == 1
    assert stats["s"].ndim == 1
