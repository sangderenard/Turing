import math

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autograd import autograd
from src.common.tensors.autoautograd.fluxspring import register_param_wheels
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    NodeSpec,
    EdgeSpec,
    FluxSpringSpec,
    SpectralCfg,
    SpectralMetrics,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    LearnCtrl,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick
from src.common.tensors.autoautograd.fluxspring.spectral_readout import compute_metrics


def _build_spec(cfg):
    node0 = NodeSpec(
        id=0,
        p0=AT.zeros(1),
        v0=AT.zeros(1),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(True, True, True)),
        scripted_axes=[0],
    )
    node1 = NodeSpec(
        id=1,
        p0=AT.zeros(1),
        v0=AT.zeros(1),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(True, True, True)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=1,
        transport=EdgeTransport(
            kappa=AT.tensor(1.0),
            learn=EdgeTransportLearn(kappa=False, k=False, l0=False, lambda_s=False, x=False),
        ),
        ctrl=EdgeCtrl(learn=LearnCtrl(True, True, True)),
    )
    spec = FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node0, node1],
        edges=[edge],
        faces=[],
        dec=DECSpec(D0=[[-1.0, 1.0]], D1=[]),
        spectral=cfg,
        gamma=AT.tensor(0.0),
    )
    wheels = register_param_wheels(spec, slots=1)
    for w in wheels:
        w.rotate(); w.bind_slot()
    edge_w = spec.edges[0].ctrl.w
    node_w = spec.nodes[1].ctrl.w
    assert all(p.requires_grad for p in (edge_w, node_w))
    return spec, edge_w, node_w


def test_spectral_fluxspring_grad():
    N = 16
    tick_hz = 160.0
    freq = 20.0
    cfg = SpectralCfg(
        enabled=False,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="hann",
        metrics=SpectralMetrics(bands=[[10.0, 30.0]]),
    )
    spec, edge_w, node_w = _build_spec(cfg)

    signal = AT.tensor([math.sin(2.0 * math.pi * freq * i / tick_hz) for i in range(N)])
    signal.requires_grad_(True)
    cfg_metrics = SpectralCfg(
        enabled=True,
        tick_hz=tick_hz,
        win_len=N,
        hop_len=N,
        window="hann",
        metrics=SpectralMetrics(bands=[[10.0, 30.0]]),
    )
    metrics = compute_metrics(signal, cfg_metrics)

    # Run a few pump ticks to exercise the dynamics.
    psi = AT.zeros(len(spec.nodes), dtype=float)
    for _ in range(N):
        psi, _ = pump_tick(psi, spec, eta=0.1, external={0: AT.tensor(1.0)})

    # Combine spectral metrics with simple algebraic terms to form a loss that
    # depends on the signal and the learnable edge/node weights.
    loss = AT.sum(metrics["bandpower"]) + AT.sum(signal * signal) + edge_w * node_w

    grads = autograd.grad(loss, [signal, edge_w, node_w])
    assert all(g is not None for g in grads)
