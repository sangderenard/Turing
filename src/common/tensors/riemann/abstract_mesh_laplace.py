"""AbstractTensor cotangent Laplacian with host-only topology assembly."""

from __future__ import annotations

import numpy as np

from ..abstraction import AbstractTensor


def abstract_mesh_laplace(
    vertices,
    triangles: np.ndarray,
    scalar_values,
    *,
    degeneracy_epsilon: float = 1e-24,
) -> AbstractTensor:
    """Apply a cotangent Laplacian using AbstractTensor numeric arithmetic.

    Integer connectivity is intentionally assembled on the host. Selection,
    edge geometry, cotangents, mass, and operator application remain tensor
    operations and therefore retain the selected AbstractTensor backend.
    """
    vertex_tensor = (
        vertices
        if isinstance(vertices, AbstractTensor)
        else AbstractTensor.tensor(vertices)
    )
    value_tensor = (
        scalar_values
        if isinstance(scalar_values, AbstractTensor)
        else AbstractTensor.tensor(scalar_values)
    )
    triangles = np.asarray(triangles, dtype=np.int64)
    dtype = vertex_tensor.get_dtype()
    device = vertex_tensor.get_device()
    backend = type(vertex_tensor)
    vertex_count = int(vertex_tensor.shape[0])
    points = vertex_tensor[triangles]
    edge_01 = points[:, 1] - points[:, 0]
    edge_02 = points[:, 2] - points[:, 0]
    gram = (
        (edge_01 * edge_01).sum(dim=1)
        * (edge_02 * edge_02).sum(dim=1)
        - (edge_01 * edge_02).sum(dim=1) ** 2
    )
    double_area = (gram + degeneracy_epsilon) ** 0.5
    triangle_area = 0.5 * double_area
    cotangents = []
    for corner, other_a, other_b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a = points[:, other_a] - points[:, corner]
        b = points[:, other_b] - points[:, corner]
        cotangents.append((a * b).sum(dim=1) / double_area)
    raw_weights = 0.5 * AbstractTensor.cat(cotangents, dim=0)

    raw_edges = np.concatenate((
        triangles[:, (1, 2)],
        triangles[:, (2, 0)],
        triangles[:, (0, 1)],
    ))
    raw_edges.sort(axis=1)
    edges, inverse = np.unique(raw_edges, axis=0, return_inverse=True)
    # The following reductions are the explicit discrete-topology boundary:
    # AbstractTensor computes every geometric coefficient and edge flux; host
    # indexed addition realizes those values on an irregular connectivity map.
    raw_weights_host = np.asarray(raw_weights.tolist(), dtype=np.float64)
    weights_host = np.zeros(len(edges), dtype=np.float64)
    np.add.at(weights_host, inverse, raw_weights_host)
    weights = AbstractTensor.tensor(
        weights_host, dtype=dtype, device=device
    ).to_backend(vertex_tensor)
    edge_delta = value_tensor[edges[:, 1]] - value_tensor[edges[:, 0]]
    flux_host = np.asarray((weights * edge_delta).tolist(), dtype=np.float64)
    numerator_host = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(numerator_host, edges[:, 0], flux_host)
    np.add.at(numerator_host, edges[:, 1], -flux_host)

    triangle_area_host = np.asarray(triangle_area.tolist(), dtype=np.float64)
    vertex_area_host = np.zeros(vertex_count, dtype=np.float64)
    for corner in range(3):
        np.add.at(
            vertex_area_host,
            triangles[:, corner],
            triangle_area_host / 3.0,
        )
    return (
        AbstractTensor.tensor(
            numerator_host, dtype=dtype, device=device
        ).to_backend(vertex_tensor)
        / AbstractTensor.tensor(
            vertex_area_host, dtype=dtype, device=device
        ).to_backend(vertex_tensor)
    )
