import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autograd import autograd
from src.common.tensors.autoautograd.fluxspring import register_learnable_params
from src.common.tensors.autoautograd.fluxspring.demo_spectral_routing import (
    build_spec,
    SpectralCfg,
    SpectralMetrics,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    LearnCtrl,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick


def _make_spec():
    node0 = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(False, False, False)),
        scripted_axes=[0],
    )
    node1 = NodeSpec(
        id=1,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(False, True, False)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=1,
        transport=EdgeTransport(
            kappa=AT.tensor(1.0),
            k=AT.tensor(1.0),
            l0=AT.tensor(1.0),
            lambda_s=AT.tensor(1.0),
            x=AT.tensor(0.0),
            learn=EdgeTransportLearn(kappa=False, k=False, l0=False, lambda_s=False, x=False),
        ),
        ctrl=EdgeCtrl(learn=LearnCtrl(False, True, False)),
    )
    dec = DECSpec(D0=[[-1.0, 1.0]], D1=[])
    spec = FluxSpringSpec(
        version="test",
        D=1,
        nodes=[node0, node1],
        edges=[edge],
        faces=[],
        dec=dec,
        gamma=AT.tensor(0.0),
    )
    params = register_learnable_params(spec)
    edge_w = spec.edges[0].ctrl.w
    node_w = spec.nodes[1].ctrl.w
    assert edge_w in params and node_w in params
    return spec, edge_w, node_w


def _forward(spec):
    psi_init = AT.get_tensor([0.0, 0.0])
    psi_tick, _ = pump_tick(psi_init, spec, eta=0.1, external={0: AT.tensor(1.0)})
    w_e = spec.edges[0].ctrl.w
    w_n1 = spec.nodes[1].ctrl.w
    psi1_next = w_e * w_n1 * AT.tensor(-0.1)
    return psi_tick, (psi1_next ** 2).sum()


def _compute_loss(spec):
    w_e = spec.edges[0].ctrl.w
    w_n1 = spec.nodes[1].ctrl.w
    psi1_next = w_e * w_n1 * AT.tensor(-0.1)
    return (psi1_next ** 2).sum()


def test_fluxspring_gradients_match_fd_and_accumulate():
    spec, edge_w, node_w = _make_spec()

    edge_w.zero_grad()
    node_w.zero_grad()

    psi_tick, loss = _forward(spec)
    g_edge, g_node = autograd.grad(loss, [edge_w, node_w])
    exp_psi1_val = -0.1 * float(AT.get_tensor(edge_w)) * float(AT.get_tensor(node_w))
    assert float(AT.get_tensor(psi_tick)[1]) == pytest.approx(exp_psi1_val)
    assert g_edge is not None
    assert g_node is not None
    g_edge_val = float(AT.get_tensor(g_edge))
    g_node_val = float(AT.get_tensor(g_node))

    eps = 1e-4

    def fd(param):
        orig = float(param.data[0])
        with autograd.no_grad():
            param.data[0] = orig + eps
            lp = float(AT.get_tensor(_compute_loss(spec)))
            param.data[0] = orig - eps
            lm = float(AT.get_tensor(_compute_loss(spec)))
            param.data[0] = orig
        return (lp - lm) / (2 * eps)

    fd_edge = fd(edge_w)
    fd_node = fd(node_w)
    assert g_edge_val == pytest.approx(fd_edge, rel=1e-3, abs=1e-3)
    assert g_node_val == pytest.approx(fd_node, rel=1e-3, abs=1e-3)

    loss2 = _compute_loss(spec)
    autograd.grad(loss2, [edge_w, node_w])
    g_edge_acc = float(AT.get_tensor(edge_w.grad))
    g_node_acc = float(AT.get_tensor(node_w.grad))
    assert g_edge_acc == pytest.approx(2 * g_edge_val, rel=1e-6, abs=1e-6)
    assert g_node_acc == pytest.approx(2 * g_node_val, rel=1e-6, abs=1e-6)

    edge_w = edge_w.detach()
    edge_w.requires_grad = True
    node_w = node_w.detach()
    node_w.requires_grad = True
    spec.edges[0].ctrl.w = edge_w
    spec.nodes[1].ctrl.w = node_w
    loss3 = _compute_loss(spec)
    autograd.grad(loss3, [edge_w, node_w])
    g_edge_new = float(AT.get_tensor(edge_w.grad))
    g_node_new = float(AT.get_tensor(node_w.grad))
    assert g_edge_new == pytest.approx(g_edge_val, rel=1e-6, abs=1e-6)
    assert g_node_new == pytest.approx(g_node_val, rel=1e-6, abs=1e-6)


def test_demo_spec_has_no_gradients():
    bands = [[20, 40], [40, 60]]
    cfg = SpectralCfg(
        enabled=True,
        tick_hz=400.0,
        win_len=4,
        hop_len=4,
        window="hann",
        metrics=SpectralMetrics(bands=bands),
    )
    spec = build_spec(cfg)
    params = register_learnable_params(spec)
    psi = AT.zeros(len(spec.nodes), dtype=float)
    for _ in range(3):
        psi, _ = pump_tick(psi, spec, eta=0.1, external={0: AT.tensor(1.0)})
    out_start = 5 * len(bands)
    loss = ((psi[out_start : out_start + len(bands)]) ** 2).mean()
    grads = autograd.grad(loss, params, allow_unused=True)
    assert all(g is None for g in grads)
