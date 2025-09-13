import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring.fs_dec import (
    dec_energy_and_gradP_AT,
    edge_params_AT,
    face_params_AT,
    edge_vectors_AT,
    _edge_indices,
    incidence_tensors_AT,
    curvature_activation_AT,
)
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
    LearnCtrl,
)
import numpy as np

def dec_energy_and_gradP_loop(P, spec: FluxSpringSpec):
    k, l0 = edge_params_AT(spec)
    k = k.reshape(-1)
    l0 = l0.reshape(-1)
    alpha, c = face_params_AT(spec)
    alpha = alpha.reshape(-1)
    c = c.reshape(-1)
    idx_i, idx_j = _edge_indices(spec)
    d = edge_vectors_AT(P, spec)
    L = AT.linalg.norm(d, dim=1) + 1e-12
    uhat = d / L.reshape(-1, 1)
    g = L - l0
    D0, D1 = incidence_tensors_AT(spec)
    if D1.shape[0] == 0:
        u = AT.zeros(0, dtype=float)
        dphi = AT.zeros(0, dtype=float)
        r = k * g
    else:
        z = D1 @ g
        u, dphi = curvature_activation_AT(z, alpha)
        r = k * g + (D1.T() @ (c * u * dphi)).reshape(-1)
    gradP = AT.zeros_like(P)
    for eidx in range(len(spec.edges)):
        i = int(idx_i[eidx])
        j = int(idx_j[eidx])
        gradP[i] += -r[eidx] * uhat[eidx]
        gradP[j] += +r[eidx] * uhat[eidx]
    E = 0.5 * AT.sum(k * g * g)
    if alpha.shape[0] > 0:
        E += 0.5 * AT.sum(c * u * u)
    return E, gradP


def _make_spec() -> FluxSpringSpec:
    node0 = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(True, True, True)),
        scripted_axes=[0],
    )
    node1 = NodeSpec(
        id=1,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(learn=LearnCtrl(True, True, True)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=1,
        transport=EdgeTransport(k=AT.tensor(1.0), l0=AT.tensor(1.0)),
        ctrl=EdgeCtrl(),
    )
    dec = DECSpec(D0=[[-1.0, 1.0]], D1=np.zeros((0, 1)).tolist())
    return FluxSpringSpec(
        version="test",
        D=1,
        nodes=[node0, node1],
        edges=[edge],
        faces=[],
        dec=dec,
        gamma=AT.tensor(0.0),
    )


def test_vectorized_grad_matches_loop():
    spec = _make_spec()
    P = AT.get_tensor([[0.0], [1.5]])
    E_vec, grad_vec = dec_energy_and_gradP_AT(P, spec)
    E_loop, grad_loop = dec_energy_and_gradP_loop(P, spec)
    assert float(E_vec) == pytest.approx(float(E_loop))
    assert AT.allclose(grad_vec, grad_loop)
