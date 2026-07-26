"""Differentiable AbstractTensor execution over shared cotangent topology."""

from __future__ import annotations

import numpy as np

from ..abstraction import AbstractTensor
from .mesh_laplace import build_cotangent_topology


def _segment_sum(values, segment_ids, segment_count: int, zero_per_segment):
    """Sum irregular segments with sort/index/cumsum AbstractTensor primitives.

    ``segment_ids`` is host topology only. Appending one tensor-valued zero for
    every segment guarantees dense output ordering without scatter assignment,
    including isolated mesh vertices.
    """

    segment_ids = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    segment_count = int(segment_count)
    if segment_count == 0:
        return zero_per_segment
    all_ids = np.concatenate(
        (segment_ids, np.arange(segment_count, dtype=np.int64))
    )
    order = np.argsort(all_ids, kind="stable")
    ordered_ids = all_ids[order]
    end_positions = np.flatnonzero(
        np.r_[ordered_ids[1:] != ordered_ids[:-1], True]
    )
    combined = AbstractTensor.cat((values, zero_per_segment), dim=0)
    cumulative = combined[order].cumsum(dim=0)
    ends = cumulative[end_positions]
    previous = AbstractTensor.cat(
        (ends[:1] * 0.0, cumulative[end_positions[:-1]]), dim=0
    )
    return ends - previous


def abstract_mesh_laplace(
    vertices,
    triangles: np.ndarray,
    scalar_values,
    *,
    degeneracy_epsilon: float = 1e-12,
    invalid_value: float = np.nan,
) -> AbstractTensor:
    """Apply the established cotangent Laplacian without numeric host breaks.

    Triangle and edge ids are discrete host topology. Geometry, cotangents,
    edge aggregation, flux divergence, lumped mass, degeneracy masking, and
    division remain AbstractTensor operations on the selected backend.
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
    vertex_count = int(vertex_tensor.shape[0])
    topology = build_cotangent_topology(triangles, vertex_count)
    # Backends such as Torch may construct an index tensor from these arrays;
    # give them writable views while the cached topology itself stays immutable.
    triangles = topology.triangles.copy()
    if tuple(value_tensor.shape) != (vertex_count,):
        raise ValueError("scalar_values needs one value per mesh vertex")
    if not len(triangles):
        return value_tensor * 0.0 + invalid_value

    points = vertex_tensor[triangles]
    edge_01 = points[:, 1] - points[:, 0]
    edge_02 = points[:, 2] - points[:, 0]
    gram = (
        (edge_01 * edge_01).sum(dim=1)
        * (edge_02 * edge_02).sum(dim=1)
        - (edge_01 * edge_02).sum(dim=1) ** 2
    )
    double_area = gram.maximum(0.0) ** 0.5
    valid_triangle = double_area > degeneracy_epsilon
    safe_double_area = AbstractTensor.where(
        valid_triangle, double_area, double_area * 0.0 + 1.0
    )
    triangle_area = AbstractTensor.where(
        valid_triangle, 0.5 * double_area, double_area * 0.0
    )

    cotangents = []
    for corner, other_a, other_b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a = points[:, other_a] - points[:, corner]
        b = points[:, other_b] - points[:, corner]
        cotangent = (a * b).sum(dim=1) / safe_double_area
        cotangents.append(
            AbstractTensor.where(
                valid_triangle, cotangent, cotangent * 0.0
            )
        )
    raw_weights = 0.5 * AbstractTensor.cat(cotangents, dim=0)
    edge_count = len(topology.edges)
    weights = _segment_sum(
        raw_weights,
        topology.edge_inverse,
        edge_count,
        raw_weights[:edge_count] * 0.0,
    )

    edges = topology.edges.copy()
    edge_delta = value_tensor[edges[:, 1]] - value_tensor[edges[:, 0]]
    flux = weights * edge_delta
    numerator = _segment_sum(
        AbstractTensor.cat((flux, -flux), dim=0),
        np.concatenate((edges[:, 0], edges[:, 1])),
        vertex_count,
        value_tensor * 0.0,
    )

    area_contributions = AbstractTensor.cat(
        (triangle_area, triangle_area, triangle_area), dim=0
    ) / 3.0
    vertex_area = _segment_sum(
        area_contributions,
        np.concatenate(
            (triangles[:, 0], triangles[:, 1], triangles[:, 2])
        ),
        vertex_count,
        value_tensor * 0.0,
    )
    degenerate_contributions = AbstractTensor.cat(
        (
            1.0 - valid_triangle,
            1.0 - valid_triangle,
            1.0 - valid_triangle,
        ),
        dim=0,
    )
    degenerate_vertices = _segment_sum(
        degenerate_contributions,
        np.concatenate(
            (triangles[:, 0], triangles[:, 1], triangles[:, 2])
        ),
        vertex_count,
        value_tensor * 0.0,
    )

    nonmanifold = np.zeros(vertex_count, dtype=np.float64)
    if np.any(topology.nonmanifold_edge_mask):
        nonmanifold[
            topology.edges[topology.nonmanifold_edge_mask].reshape(-1)
        ] = 1.0
    nonmanifold_tensor = AbstractTensor.tensor(
        nonmanifold,
        dtype=vertex_tensor.get_dtype(),
        device=vertex_tensor.get_device(),
    ).to_backend(vertex_tensor)
    valid_vertex = (
        (vertex_area > degeneracy_epsilon)
        * (degenerate_vertices == 0.0)
        * (nonmanifold_tensor == 0.0)
    )
    safe_vertex_area = AbstractTensor.where(
        vertex_area > degeneracy_epsilon,
        vertex_area,
        vertex_area * 0.0 + 1.0,
    )
    laplacian = numerator / safe_vertex_area
    return AbstractTensor.where(
        valid_vertex,
        laplacian,
        laplacian * 0.0 + invalid_value,
    )
