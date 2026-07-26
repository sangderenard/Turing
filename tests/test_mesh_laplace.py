import numpy as np

from src.common.tensors.riemann.mesh_laplace import mesh_laplace_beltrami


def _grid():
    vertices = np.asarray([
        (x, y, 0.0, 0.0, 0.0)
        for x in (0.0, 0.5, 1.0)
        for y in (0.0, 0.5, 1.0)
    ])
    triangles = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            b = (i + 1) * 3 + j
            c = (i + 1) * 3 + j + 1
            d = i * 3 + j + 1
            triangles.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(triangles)


def test_full_embedding_cotangent_laplace_is_exact_at_planar_interior():
    vertices, triangles = _grid()
    scalar = vertices[:, 0] ** 2 + vertices[:, 1] ** 2
    result = mesh_laplace_beltrami(vertices, triangles, scalar)
    assert np.isclose(result.laplacian[4], 4.0)
    assert not result.geometry.boundary_vertex_mask[4]
    assert not result.geometry.singular_vertex_mask.any()


def test_degenerate_triangle_is_reported_without_silent_division():
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    result = mesh_laplace_beltrami(
        vertices, np.asarray(((0, 1, 2),)), np.asarray((0.0, 1.0, 2.0))
    )
    assert result.geometry.degenerate_triangle_mask[0]
    assert result.geometry.singular_vertex_mask.all()
    assert np.isnan(result.laplacian).all()


def test_nonmanifold_vertices_are_invalidated():
    vertices = np.asarray((
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.0, -1.0), (0.5, 0.5),
    ))
    triangles = np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)))
    result = mesh_laplace_beltrami(
        vertices, triangles, np.arange(len(vertices), dtype=float)
    )
    assert result.geometry.nonmanifold_edge_mask.any()
    assert result.geometry.nonmanifold_vertex_mask[[0, 1]].all()
    assert np.isnan(result.laplacian[[0, 1]]).all()


def test_geometry_arrays_are_immutable():
    vertices, triangles = _grid()
    result = mesh_laplace_beltrami(vertices, triangles, vertices[:, 0])
    assert not result.geometry.edges.flags.writeable
    assert not result.geometry.lumped_vertex_areas.flags.writeable
