import numpy as np

from src.common.tensors import AbstractTensor
from src.common.tensors.riemann import abstract_mesh_laplace


def test_abstract_cotangent_laplace_is_exact_on_planar_interior():
    vertices = np.asarray([
        (x, y, 0.0, 0.0) for x in (0.0, 0.5, 1.0)
        for y in (0.0, 0.5, 1.0)
    ])
    triangles = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            triangles.extend(((a, a + 3, a + 4), (a, a + 4, a + 1)))
    scalar = vertices[:, 0] ** 2 + vertices[:, 1] ** 2
    result = abstract_mesh_laplace(
        AbstractTensor.tensor(vertices), np.asarray(triangles),
        AbstractTensor.tensor(scalar),
    )
    assert isinstance(result, AbstractTensor)
    assert np.isclose(result.tolist()[4], 4.0)
