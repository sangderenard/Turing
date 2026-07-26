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


def test_c_backend_basic_slice_assignment_uses_shared_index_policy():
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor(np.arange(24.0).reshape(2, 3, 4))
        values[..., 1, 1:4:2] = AbstractTensor.tensor(
            [[101.0, 102.0], [201.0, 202.0]]
        )
        values[0, -1, :] = -3.0

    expected = np.arange(24.0).reshape(2, 3, 4)
    expected[..., 1, 1:4:2] = [[101.0, 102.0], [201.0, 202.0]]
    expected[0, -1, :] = -3.0
    assert values.tolist() == expected.tolist()


def test_c_backend_runs_composite_linalg_solve():
    matrix_values = np.asarray(
        [[4.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]]
    )
    rhs_values = np.asarray([[2.0], [4.0], [5.0]])
    with AbstractTensor.use_backend("c"):
        matrix = AbstractTensor.tensor(matrix_values)
        rhs = AbstractTensor.tensor(rhs_values)
        solved = AbstractTensor.linalg.solve(matrix, rhs)

    assert np.allclose(matrix_values @ np.asarray(solved.tolist()), rhs_values)


def test_c_backend_slice_backward_uses_native_index_assignment():
    with AbstractTensor.use_backend("c"):
        values = AbstractTensor.tensor(np.arange(12.0).reshape(3, 4))
        values.requires_grad_(True)
        loss = values[1:, 1:3].sum()
        gradient = AbstractTensor.autograd.grad(loss, [values])[0]

    assert gradient.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
