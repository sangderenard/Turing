import numpy as np

from src.common.tensors.riemann import (
    AdaptiveSurfaceTriangulator,
    TriangulationTolerance,
)


def _paraboloid(parameters):
    u, v = parameters.T
    return np.stack((u, v, u * u + 0.5 * v * v), axis=1)


def _paraboloid_jacobian(parameters):
    u, v = parameters.T
    result = np.zeros((len(parameters), 3, 2))
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = 1.0
    result[:, 2, 0] = 2.0 * u
    result[:, 2, 1] = v
    return result


def _mesh_edges(triangles):
    return {
        tuple(sorted(edge))
        for triangle in triangles
        for edge in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    }


def test_curved_surface_refines_in_parallel_generations():
    result = AdaptiveSurfaceTriangulator(
        _paraboloid,
        jacobian=_paraboloid_jacobian,
        tolerance=TriangulationTolerance(
            position=0.02, tangent=0.2, max_rounds=8
        ),
        initial_resolution=(1, 1),
    ).triangulate()
    assert result.converged
    assert result.generation > 0
    assert result.triangle_count > 2
    assert result.position_error.max() <= 0.02
    assert result.tangent_error is not None
    assert result.tangent_error.max() <= 0.2


def test_refinement_has_no_hanging_full_edges():
    result = AdaptiveSurfaceTriangulator(
        _paraboloid,
        tolerance=TriangulationTolerance(position=0.03, max_rounds=8),
    ).triangulate()
    edges = _mesh_edges(result.triangles)
    for a, b in edges:
        midpoint = (result.parameters[a] + result.parameters[b]) * 0.5
        existing = np.flatnonzero(np.all(
            np.isclose(result.parameters, midpoint, atol=1e-12), axis=1
        ))
        assert not any(
            vertex not in (a, b)
            and tuple(sorted((a, vertex))) in edges
            and tuple(sorted((vertex, b))) in edges
            for vertex in existing
        )


def test_flat_surface_needs_no_refinement():
    flat = lambda uv: np.column_stack((uv, np.zeros(len(uv))))
    result = AdaptiveSurfaceTriangulator(
        flat, tolerance=TriangulationTolerance(position=1e-8)
    ).triangulate()
    assert result.converged
    assert result.generation == 0
    assert result.triangle_count == 2


def test_tangent_certificate_compares_against_piecewise_linear_tangent():
    result = AdaptiveSurfaceTriangulator(
        _paraboloid,
        jacobian=_paraboloid_jacobian,
        tolerance=TriangulationTolerance(
            position=100.0, tangent=0.3, max_rounds=8
        ),
        initial_resolution=(1, 1),
    ).triangulate()
    assert result.generation > 0
    assert result.tangent_error is not None
    assert result.tangent_error.max() <= 0.3


def test_interior_probes_break_edge_midpoint_aliasing():
    def aliased(parameters):
        u, v = parameters.T
        return np.column_stack((u, v, np.sin(4 * np.pi * u) * np.sin(4 * np.pi * v)))

    result = AdaptiveSurfaceTriangulator(
        aliased,
        tolerance=TriangulationTolerance(position=0.1, max_rounds=3),
        initial_resolution=(1, 1),
    ).triangulate()
    assert result.generation > 0
    assert result.position_error.max() > 0.1


def test_triangle_budget_is_a_strict_upper_bound():
    result = AdaptiveSurfaceTriangulator(
        _paraboloid,
        tolerance=TriangulationTolerance(
            position=1e-12, max_rounds=8, max_triangles=5
        ),
        initial_resolution=(1, 1),
    ).triangulate()
    assert not result.converged
    assert result.triangle_count <= 5


def test_results_own_immutable_array_storage():
    result = AdaptiveSurfaceTriangulator(_paraboloid).triangulate()
    assert not result.parameters.flags.writeable
    assert not result.embedded.flags.writeable
    assert not result.triangles.flags.writeable


def test_callable_output_shapes_are_checked():
    bad = lambda uv: np.zeros((len(uv), 3, 1))
    with np.testing.assert_raises(ValueError):
        AdaptiveSurfaceTriangulator(bad).triangulate()
