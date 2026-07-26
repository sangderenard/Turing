import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.autograd import GradTape, autograd
from src.common.tensors.riemann import (
    abstract_mesh_laplace,
    mesh_laplace_beltrami,
)


def _grid():
    vertices = np.asarray([
        (x, y, 0.0, 0.0) for x in (0.0, 0.5, 1.0)
        for y in (0.0, 0.5, 1.0)
    ])
    triangles = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            triangles.extend(((a, a + 3, a + 4), (a, a + 4, a + 1)))
    return vertices, np.asarray(triangles)


def test_abstract_cotangent_laplace_is_exact_on_planar_interior():
    vertices, triangles = _grid()
    scalar = vertices[:, 0] ** 2 + vertices[:, 1] ** 2
    result = abstract_mesh_laplace(
        AbstractTensor.tensor(vertices), triangles,
        AbstractTensor.tensor(scalar),
    )
    assert isinstance(result, AbstractTensor)
    assert np.isclose(result.tolist()[4], 4.0)


def test_abstract_path_matches_established_mesh_dec_operator():
    vertices, triangles = _grid()
    scalar = (
        vertices[:, 0] ** 2
        + 0.25 * vertices[:, 0] * vertices[:, 1]
        - 0.5 * vertices[:, 1]
    )
    expected = mesh_laplace_beltrami(
        vertices, triangles, scalar
    ).laplacian
    actual = np.asarray(
        abstract_mesh_laplace(
            AbstractTensor.tensor(vertices),
            triangles,
            AbstractTensor.tensor(scalar),
        ).tolist()
    )
    np.testing.assert_allclose(actual, expected)


def test_abstract_path_preserves_degenerate_and_nonmanifold_masks():
    degenerate_vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    )
    degenerate_triangles = np.asarray(((0, 1, 2),))
    degenerate = abstract_mesh_laplace(
        AbstractTensor.tensor(degenerate_vertices),
        degenerate_triangles,
        AbstractTensor.tensor((0.0, 1.0, 2.0)),
    )
    assert np.isnan(np.asarray(degenerate.tolist())).all()

    nonmanifold_vertices = np.asarray((
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.0, -1.0), (0.5, 0.5),
    ))
    nonmanifold_triangles = np.asarray(
        ((0, 1, 2), (1, 0, 3), (0, 1, 4))
    )
    nonmanifold = np.asarray(
        abstract_mesh_laplace(
            AbstractTensor.tensor(nonmanifold_vertices),
            nonmanifold_triangles,
            AbstractTensor.tensor(np.arange(5.0)),
        ).tolist()
    )
    assert np.isnan(nonmanifold[[0, 1]]).all()


@pytest.mark.parametrize("backend", ["numpy", "c", "torch"])
def test_abstract_mesh_laplace_keeps_value_and_geometry_gradients(backend):
    vertices, triangles = _grid()
    autograd.tape = GradTape()
    with AbstractTensor.use_backend(backend):
        vertex_tensor = AbstractTensor.tensor(vertices)
        vertex_tensor.requires_grad_(True)
        values = AbstractTensor.tensor(
            vertices[:, 0] ** 2 + vertices[:, 1] ** 2
        )
        values.requires_grad_(True)
        laplacian = abstract_mesh_laplace(vertex_tensor, triangles, values)
        loss = (laplacian * laplacian).sum()
        vertex_gradient, value_gradient = AbstractTensor.autograd.grad(
            loss, [vertex_tensor, values]
        )
    assert np.isfinite(np.asarray(vertex_gradient.tolist())).all()
    assert np.isfinite(np.asarray(value_gradient.tolist())).all()
    assert np.any(np.asarray(vertex_gradient.tolist()) != 0.0)
    assert np.any(np.asarray(value_gradient.tolist()) != 0.0)
