"""``DECSystem``: the exterior calculus of a complex, checked against invariants.

This realizes the contract ``geometry/src/geometry/dec_toolkit_header.cpp``
declared and never implemented.  It is checked three ways, none of which is
a tolerance on a number somebody chose:

* against an INDEPENDENT implementation -- ``laplace0`` must reproduce
  ``CotangentMeshGeometry.apply``, which computes the same operator by an
  entirely different route and was already in the tree and tested;
* against ``d . d == 0``, the identity that makes a complex a complex;
* against the HODGE THEOREM, ``dim ker Delta_k == b_k``.  A sphere has
  ``b0, b1 = 1, 0`` and a torus has ``1, 2``, and an operator either
  reproduces those integers or it is not a Laplace-de Rham operator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.abstract_convolution.dec_system import DECSystem


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def _octahedron():
    """A closed surface of genus 0: ``chi = 2``, ``b0, b1 = 1, 0``."""
    vertices = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0],
                         [0, -1, 0], [0, 0, 1.], [0, 0, -1]])
    triangles = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                          [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return vertices, triangles


def _torus(n=8, m=8, radius=3.0, tube=1.0, skew=0.5):
    """Genus 1: ``chi = 0``, ``b0, b1 = 1, 2``.

    Rows are offset by half a cell so the quads are RHOMBIC.  Split a
    rectangle instead and every diagonal faces two right angles, whose
    cotangents are zero -- see ``test_a_vanishing_star1_is_refused``.
    """
    node = lambda i, j: (i % n) * m + (j % m)
    vertices = np.zeros((n * m, 3))
    for i in range(n):
        for j in range(m):
            u = 2 * np.pi * ((i + skew * j) % n) / n
            v = 2 * np.pi * j / m
            vertices[node(i, j)] = [
                (radius + tube * np.cos(v)) * np.cos(u),
                (radius + tube * np.cos(v)) * np.sin(u),
                tube * np.sin(v),
            ]
    triangles = []
    for i in range(n):
        for j in range(m):
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i + 1, j + 1), node(i, j + 1)
            triangles += [[a, b, c], [a, c, d]]
    return vertices, np.array(triangles)


def _rectangular_torus(n=6, m=6):
    """The same surface with RIGHT-angled quads: ``*1`` vanishes on the
    diagonals, which is the degeneracy the system refuses."""
    return _torus(n=n, m=m, skew=0.0)


def _kernel_dimension(matrix, tol=1e-8):
    """Symmetry-free: these operators are self-adjoint in ``*k``, not
    symmetric, so an eigenvalue count would need the right inner product
    while a rank does not."""
    return matrix.shape[1] - np.linalg.matrix_rank(matrix, tol=tol)


SURFACES = [
    pytest.param(_octahedron, 2, 1, 0, id="sphere"),
    pytest.param(_torus, 0, 1, 2, id="torus"),
]


# --------------------------------------------------------------------------
# the complex
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_euler_characteristic_of_the_surface(build, characteristic, b0, b1) -> None:
    system = DECSystem.from_triangle_mesh(*build())

    assert system.complex.euler_characteristic == characteristic


@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_boundary_of_boundary_vanishes(build, characteristic, b0, b1) -> None:
    """``d1 @ d0 == 0`` on the triangles, not only on detected faces."""
    system = DECSystem.from_triangle_mesh(*build())

    composed = system.matrix("d1") @ system.matrix("d0")
    assert np.abs(composed).max() < 1e-12


@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_curl_of_a_gradient_is_zero(build, characteristic, b0, b1) -> None:
    """The same identity in the names the header gave the operators."""
    system = DECSystem.from_triangle_mesh(*build())
    scalar = np.random.default_rng(0).normal(size=system.complex.vertex_count)

    circulation = system.curl(system.gradient(AT.get_tensor(scalar)))

    assert float(AT.linalg.norm(circulation)) < 1e-12


# --------------------------------------------------------------------------
# against the independent implementation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_laplace0_matches_the_cotangent_reference(build, characteristic, b0, b1) -> None:
    """``laplace0`` reproduces ``CotangentMeshGeometry.apply``, negated.

    ``apply`` assembles the same operator without ever naming ``d0``, a
    Hodge star or a codifferential, so agreeing with it is agreement
    between two different constructions rather than a self-check.  The
    sign is the convention difference: ``apply`` is the geometer's
    ``div grad`` and ``laplace0`` is ``delta d``, which is its negative.
    """
    system = DECSystem.from_triangle_mesh(*build())
    scalar = np.random.default_rng(0).normal(size=system.complex.vertex_count)

    mine = np.asarray(system.laplace0(AT.get_tensor(scalar)).data)
    reference = -system.geometry.apply(scalar)

    assert np.allclose(mine, reference, atol=1e-12)


@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_laplace0_is_self_adjoint_in_star0(build, characteristic, b0, b1) -> None:
    """``*0 @ laplace0`` is symmetric: it is ``d0^T *1 d0``."""
    system = DECSystem.from_triangle_mesh(*build())
    mass = np.asarray(system.star0.data)

    weighted = np.diag(mass) @ system.matrix("laplace0")

    assert np.allclose(weighted, weighted.T, atol=1e-10)


def test_divergence_of_a_gradient_is_the_negated_laplacian() -> None:
    """The header's three names compose the way their meanings say."""
    system = DECSystem.from_triangle_mesh(*_octahedron())
    scalar = AT.get_tensor(
        np.random.default_rng(1).normal(size=system.complex.vertex_count))

    divergence = system.divergence(system.gradient(scalar))
    laplacian = system.laplace0(scalar)

    assert np.allclose(np.asarray(divergence.data),
                       -np.asarray(laplacian.data), atol=1e-12)


# --------------------------------------------------------------------------
# the Hodge theorem
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build,characteristic,b0,b1", SURFACES)
def test_hodge_theorem_counts_the_betti_numbers(build, characteristic, b0, b1) -> None:
    """``dim ker Delta_k == b_k`` -- a topological invariant, not a tolerance.

    ``laplace0``'s kernel is the constants, one per connected component.
    ``laplace1``'s kernel is the harmonic 1-forms: none on a sphere, and
    exactly two on a torus, one for each way round it.  Nothing about the
    mesh's resolution or shape may move these integers.
    """
    system = DECSystem.from_triangle_mesh(*build())

    assert _kernel_dimension(system.matrix("laplace0")) == b0
    assert _kernel_dimension(system.matrix("laplace1")) == b1


def test_the_harmonic_space_is_what_the_complex_predicts() -> None:
    """``dim ker d1 - dim im d0 == b1``, computed from the ranks alone.

    The same number the Hodge theorem gives above, reached without the
    metric: if these two ever disagree, the stars are wrong rather than
    the topology.
    """
    system = DECSystem.from_triangle_mesh(*_torus())
    d0 = system.matrix("d0")
    d1 = system.matrix("d1")

    cycles = system.complex.edge_count - np.linalg.matrix_rank(d1)
    boundaries = np.linalg.matrix_rank(d0)

    assert cycles - boundaries == 2
    assert _kernel_dimension(system.matrix("laplace1")) == 2


# --------------------------------------------------------------------------
# the degeneracy that hid all of this
# --------------------------------------------------------------------------

def test_a_vanishing_star1_is_refused_rather_than_divided_by() -> None:
    """Right-angled quads give zero cotangent weights, and ``delta2`` divides.

    Measured on a 6x6 rectangular torus: 42 of 108 weights within 1e-9 of
    zero, and ``laplace1`` reported NO harmonic 1-forms on a surface with
    two.  A wrong Betti number is exactly the kind of answer that looks
    like a result, so the division is refused and says why.
    """
    system = DECSystem.from_triangle_mesh(*_rectangular_torus())

    assert system.degenerate_star1.any()
    with pytest.raises(ValueError, match="vanishes on"):
        system.laplace1(AT.zeros((system.complex.edge_count,)))

    # laplace0 never divides by *1 and stays usable on the same mesh.
    scalar = np.random.default_rng(2).normal(size=system.complex.vertex_count)
    assert np.allclose(np.asarray(system.laplace0(AT.get_tensor(scalar)).data),
                       -system.geometry.apply(scalar), atol=1e-12)


def test_negative_star1_is_not_the_degenerate_case() -> None:
    """A non-Delaunay mesh has negative weights and still counts b1.

    Worth pinning because the two are easy to conflate: negative weights
    make the operator non-positive but leave the de Rham structure
    intact, and this torus has 64 of them.
    """
    system = DECSystem.from_triangle_mesh(*_torus())
    weights = np.asarray(system.star1.data)

    assert (weights < 0).any()
    assert not system.degenerate_star1.any()
    assert _kernel_dimension(system.matrix("laplace1")) == 2


# --------------------------------------------------------------------------
# the lattice: the same operator, specialising to the other textbook
# --------------------------------------------------------------------------

LATTICES = [(3, 3, 3), (4, 3, 2), (5, 4, 3)]


def _expected_cells(shape):
    nx, ny, nz = shape
    return (
        nx * ny * nz,
        (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1),
        ((nx - 1) * (ny - 1) * nz + nx * (ny - 1) * (nz - 1)
         + (nx - 1) * ny * (nz - 1)),
        (nx - 1) * (ny - 1) * (nz - 1),
    )


@pytest.mark.parametrize("shape", LATTICES)
def test_lattice_enumerates_its_own_cells(shape) -> None:
    """A lattice knows its squares; it does not rediscover them by walking.

    The counts are closed form, so this catches an off-by-one in the cell
    enumeration that a boundary identity would happily satisfy.
    """
    system = DECSystem.from_lattice(shape)
    complex = system.complex

    assert (complex.vertex_count, complex.edge_count,
            complex.face_count, complex.volume_count) == _expected_cells(shape)
    assert complex.euler_characteristic == 1


@pytest.mark.parametrize("shape", LATTICES)
def test_lattice_boundary_identities_hold_at_both_levels(shape) -> None:
    """``d1 @ d0 == 0`` and ``d2 @ d1 == 0`` on directly enumerated cells."""
    system = DECSystem.from_lattice(shape)

    assert np.abs(system.matrix("d1") @ system.matrix("d0")).max() < 1e-12
    assert np.abs(system.matrix("d2") @ system.matrix("d1")).max() < 1e-12


@pytest.mark.parametrize("spacing", [0.5, 1.0, 2.5])
def test_lattice_laplace0_is_the_seven_point_stencil(spacing) -> None:
    """The DEC Laplacian on a cubic lattice IS the finite-difference one.

    This is the reason the metric is built from circumcentric duals
    rather than chosen: with ``*0 = h^3`` and ``*1 = h``, the operator
    ``*0^-1 d0^T *1 d0`` collapses to ``(1/h^2) d0^T d0``, whose interior
    row is ``6/h^2`` on the centre and ``-1/h^2`` on each of six
    neighbours.  Nobody wrote that stencil down; it falls out.  The same
    expression on a triangle mesh gives the cotangent weights, which
    ``test_laplace0_matches_the_cotangent_reference`` pins separately.
    """
    extent = 5
    system = DECSystem.from_lattice((extent, extent, extent), spacing=spacing)
    laplacian = system.matrix("laplace0")

    vertex = lambda i, j, k: i + extent * (j + extent * k)
    centre = vertex(2, 2, 2)
    neighbours = [vertex(1, 2, 2), vertex(3, 2, 2), vertex(2, 1, 2),
                  vertex(2, 3, 2), vertex(2, 2, 1), vertex(2, 2, 3)]
    row = laplacian[centre]

    assert row[centre] == pytest.approx(6.0 / spacing ** 2)
    for neighbour in neighbours:
        assert row[neighbour] == pytest.approx(-1.0 / spacing ** 2)
    assert np.allclose(np.delete(row, [centre] + neighbours), 0.0)
    # A constant field is in the kernel, so every row sums to zero.
    assert abs(row.sum()) < 1e-12


def test_lattice_betti_numbers_of_a_solid_block() -> None:
    """A solid block is contractible: ``b0, b1 = 1, 0``.

    ``laplace1`` here is the full three-dimensional curl-curl plus
    grad-div, which is the operator the field work runs on, so its kernel
    being empty is the statement that the block traps no flux.
    """
    system = DECSystem.from_lattice((4, 4, 4))

    assert _kernel_dimension(system.matrix("laplace0")) == 1
    assert _kernel_dimension(system.matrix("laplace1")) == 0


def test_lattice_star_values_are_the_circumcentric_ratios() -> None:
    """``*k`` is dual measure over primal measure, exactly, on a lattice."""
    spacing = 0.25
    system = DECSystem.from_lattice((3, 3, 3), spacing=spacing)

    assert np.allclose(np.asarray(system.star0.data), spacing ** 3)
    assert np.allclose(np.asarray(system.star1.data), spacing)
    assert np.allclose(np.asarray(system.star2.data), 1.0 / spacing)
    assert np.allclose(np.asarray(system.star3.data), 1.0 / spacing ** 3)
