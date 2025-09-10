import pytest
from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autograd import autograd
from src.common.tensors.autoautograd.fluxspring.fs_types import (
    FluxSpringSpec,
    NodeSpec,
    EdgeSpec,
    FaceSpec,
    DECSpec,
    NodeCtrl,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    LearnCtrl,
    FaceLearn,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import (
    edge_params_AT,
    face_params_AT,
)
from src.common.tensors.autoautograd.fluxspring import register_learnable_params


def _make_spec():
    nodes = [
        NodeSpec(
            id=0,
            p0=AT.get_tensor([0.0]),
            v0=AT.get_tensor([0.0]),
            mass=AT.tensor(1.0),
            ctrl=NodeCtrl(learn=LearnCtrl(False, False, False)),
            scripted_axes=[0],
        ),
        NodeSpec(
            id=1,
            p0=AT.get_tensor([0.0]),
            v0=AT.get_tensor([0.0]),
            mass=AT.tensor(1.0),
            ctrl=NodeCtrl(learn=LearnCtrl(False, False, False)),
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
            learn=EdgeTransportLearn(kappa=False, k=True, l0=True, lambda_s=False, x=False),
        ),
        ctrl=EdgeCtrl(learn=LearnCtrl(False, False, False)),
    )
    face = FaceSpec(
        edges=[1],
        alpha=AT.tensor(0.5),
        c=AT.tensor(1.0),
        learn=FaceLearn(alpha=True, c=True),
    )
    dec = DECSpec(D0=[[-1.0, 1.0]], D1=[[0.0]])
    spec = FluxSpringSpec(
        version="test",
        D=1,
        nodes=nodes,
        edges=[edge],
        faces=[face],
        dec=dec,
        gamma=AT.tensor(0.0),
    )
    return spec


def test_edge_face_params_gradients_and_dtype():
    spec = _make_spec()
    params = register_learnable_params(spec)
    assert len(params) == 4

    k, l0 = edge_params_AT(spec)
    alpha, c = face_params_AT(spec)

    # dtype preservation
    assert k.dtype == spec.edges[0].transport.k.dtype
    assert l0.dtype == spec.edges[0].transport.l0.dtype
    assert alpha.dtype == spec.faces[0].alpha.dtype
    assert c.dtype == spec.faces[0].c.dtype

    loss = (k.sum() + l0.sum() + alpha.sum() + c.sum()) * AT.tensor(2.0)
    grads = autograd.grad(loss, params)
    assert all(g is not None for g in grads)
    for g in grads:
        assert float(AT.get_tensor(g)) == pytest.approx(2.0)
