import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.core import Linear
from src.common.tensors.abstract_nn.losses import MSELoss
from src.common.tensors.riemann import abstract_mesh_laplace


def test_c_backend_runs_universal_linalg_and_loss_compositions():
    with AbstractTensor.use_backend("c"):
        identity = AbstractTensor.eye(3)
        vector = AbstractTensor.tensor([[3.0, 4.0, 0.0]])
        norm = AbstractTensor.linalg.norm(vector, dim=1)
        loss = MSELoss().forward(identity, identity * 0.5)

    assert identity.tolist() == np.eye(3).tolist()
    assert norm.tolist() == [5.0]
    assert np.isclose(loss.item(), 1.0 / 12.0)


def test_c_backend_runs_abstract_nn_linear_layer_without_special_kernel():
    with AbstractTensor.use_backend("c"):
        inputs = AbstractTensor.tensor(
            [[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]]
        )
        layer = Linear(3, 2, like=inputs, bias=True)
        output = layer.forward(inputs)

    assert output.shape == (2, 2)
    assert np.isfinite(np.asarray(output.tolist())).all()


def test_c_backend_runs_riemannian_mesh_laplace_composition():
    vertices = np.asarray([
        (x, y, 0.0) for x in (0.0, 0.5, 1.0)
        for y in (0.0, 0.5, 1.0)
    ])
    triangles = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            triangles.extend(((a, a + 3, a + 4), (a, a + 4, a + 1)))
    scalar = vertices[:, 0] ** 2 + vertices[:, 1] ** 2

    with AbstractTensor.use_backend("c"):
        result = abstract_mesh_laplace(
            AbstractTensor.tensor(vertices),
            np.asarray(triangles),
            AbstractTensor.tensor(scalar),
        )

    assert np.isclose(result.tolist()[4], 4.0)
