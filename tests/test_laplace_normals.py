import numpy as np
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.common.tensors import AbstractTensor


def test_autograd_identity_normals():
    u = AbstractTensor.linspace(0, 1, 3)
    v = AbstractTensor.linspace(0, 1, 3)
    w = AbstractTensor.linspace(0, 1, 3)
    U, V, W = AbstractTensor.meshgrid(u, v, w, indexing='ij')

    U.requires_grad = True
    V.requires_grad = True
    W.requires_grad = True

    X, Y, Z = U, V, W

    dXdu = AbstractTensor.autograd.grad(X, [U], grad_outputs=AbstractTensor.ones_like(X), retain_graph=True)[0]
    dYdv = AbstractTensor.autograd.grad(Y, [V], grad_outputs=AbstractTensor.ones_like(Y), retain_graph=True)[0]
    dZdw = AbstractTensor.autograd.grad(Z, [W], grad_outputs=AbstractTensor.ones_like(Z), retain_graph=True)[0]

    e_u = AbstractTensor.stack([dXdu, AbstractTensor.zeros_like(dXdu), AbstractTensor.zeros_like(dXdu)], dim=-1)
    e_v = AbstractTensor.stack([AbstractTensor.zeros_like(dYdv), dYdv, AbstractTensor.zeros_like(dYdv)], dim=-1)
    normals = AbstractTensor.linalg.cross(e_u, e_v, dim=-1)

    norms = np.linalg.norm(normals, axis=-1)
    assert np.allclose(norms, np.ones_like(norms))
