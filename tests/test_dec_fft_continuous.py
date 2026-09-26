"""The DEC <-> FFT <-> continuous-Euclidean identity set.

Three constructions of "the Laplacian" live in this tree and have never been
compared to each other:

* ``BuildGraphLaplace``     -- the combinatorial ``L = D - A``.
* DEC incidence             -- ``d0^T (*1) d0``, the exterior-calculus assembly.
* ``continuous_laplace_beltrami`` -- the metric operator on a smooth manifold.

On a *periodic* domain all three provably agree, and the resulting operator is
circulant, so its spectrum is obtainable by FFT in O(N log N) instead of a
dense eigensolve.  That makes the periodic case the one place where the whole
set can be pinned against each other with exact analytic answers, which is what
this module does:

    d0^T d0  ==  D - A                     (DEC == graph)
    D1 @ D0  ==  0                          (d . d == 0)
    fft(row0(L))  ==  eigenvalues(L)        (FFT == spectrum)
    fft-symbol / h^2  ->  Beltrami spectrum (discrete -> continuous, O(h^2))

Every computation runs through ``AbstractTensor`` ops so the identities hold
for any registered backend; numpy appears only in fixture construction and in
assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.abstract_convolution.laplace_nd import (
    BuildGraphLaplace,
    continuous_laplace_beltrami,
)


# --------------------------------------------------------------------------
# fixtures: periodic complexes, built as DEC incidence matrices
# --------------------------------------------------------------------------

def _ring_d0(n: int) -> np.ndarray:
    """``d0`` for a periodic 1D ring: (E, N) with E == N.

    Edge ``e`` runs from node ``e`` to node ``e + 1 (mod n)``, so ``(d0 f)_e``
    is the forward difference of the 0-form ``f`` along that edge.
    """
    d0 = np.zeros((n, n))
    for e in range(n):
        d0[e, e] = -1.0
        d0[e, (e + 1) % n] = 1.0
    return d0


def _torus_d0_d1(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``d0`` and ``d1`` for a periodic 2D grid (a discrete torus).

    Nodes are ``n * n``; edges are one horizontal and one vertical per node;
    faces are the ``n * n`` unit plaquettes.  Orientations are chosen so that
    each face boundary traverses ``+H, +V, -H, -V`` counter-clockwise, which is
    what makes ``d1 . d0 == 0`` hold rather than merely being asserted.
    """
    node = lambda r, c: (r % n) * n + (c % n)
    edge_h = lambda r, c: (r % n) * n + (c % n)              # (r,c) -> (r,c+1)
    edge_v = lambda r, c: n * n + (r % n) * n + (c % n)      # (r,c) -> (r+1,c)

    num_nodes, num_edges, num_faces = n * n, 2 * n * n, n * n
    d0 = np.zeros((num_edges, num_nodes))
    for r in range(n):
        for c in range(n):
            h = edge_h(r, c)
            d0[h, node(r, c)] = -1.0
            d0[h, node(r, c + 1)] = 1.0
            v = edge_v(r, c)
            d0[v, node(r, c)] = -1.0
            d0[v, node(r + 1, c)] = 1.0

    d1 = np.zeros((num_faces, num_edges))
    for r in range(n):
        for c in range(n):
            f = r * n + c
            d1[f, edge_h(r, c)] = 1.0
            d1[f, edge_v(r, c + 1)] = 1.0
            d1[f, edge_h(r + 1, c)] = -1.0
            d1[f, edge_v(r, c)] = -1.0
    return d0, d1


def _ring_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[i, (i - 1) % n] = 1.0
    return adj


def _torus_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n * n, n * n))
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                adj[i, ((r + dr) % n) * n + ((c + dc) % n)] = 1.0
    return adj


def _np(t) -> np.ndarray:
    """Extract a numpy view for assertions/reporting only."""
    return np.asarray(t.data if hasattr(t, "data") else t)


# --------------------------------------------------------------------------
# leg 1: DEC assembly == graph Laplacian
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [6, 16])
def test_dec_d0_matches_graph_laplacian_ring(n: int) -> None:
    """``d0^T d0`` reproduces ``BuildGraphLaplace``'s ``D - A`` on a ring."""
    d0 = AT.get_tensor(_ring_d0(n))
    L_dec = d0.T() @ d0

    L_graph, _, _ = BuildGraphLaplace(AT.get_tensor(_ring_adjacency(n))).build()

    assert np.allclose(_np(L_dec), _np(L_graph)), (
        "DEC assembly and graph Laplacian disagree:\n"
        f"  d0^T d0 row0 = {_np(L_dec)[0]}\n"
        f"  D - A   row0 = {_np(L_graph)[0]}"
    )


def test_dec_d0_matches_graph_laplacian_torus() -> None:
    """Same identity on the 2D periodic complex, where faces also exist."""
    n = 6
    d0_np, _ = _torus_d0_d1(n)
    d0 = AT.get_tensor(d0_np)
    L_dec = d0.T() @ d0

    L_graph, _, _ = BuildGraphLaplace(AT.get_tensor(_torus_adjacency(n))).build()

    assert np.allclose(_np(L_dec), _np(L_graph))
    # Degree 4 everywhere: the torus is a regular complex, which is exactly the
    # constant-diagonal case that used to defeat the Jacobi eigensolver.
    assert np.allclose(np.diag(_np(L_dec)), 4.0)


def test_dec_hodge_star_gives_weighted_laplacian() -> None:
    """``d0^T (*1) d0`` with a diagonal Hodge star is the *weighted* Laplacian.

    This is the generalisation the plain incidence identity is a special case
    of: the 1-form Hodge star carries the edge metric, so varying it must
    reproduce ``BuildGraphLaplace`` on a weighted adjacency.
    """
    n = 8
    rng = np.random.default_rng(0)
    weights = rng.random(n) + 0.5                    # one weight per ring edge

    d0 = AT.get_tensor(_ring_d0(n))
    hodge1 = AT.get_tensor(np.diag(weights))         # *1 on 1-forms
    L_dec = d0.T() @ (hodge1 @ d0)

    adj = np.zeros((n, n))
    for e in range(n):                               # edge e joins e and e+1
        adj[e, (e + 1) % n] = weights[e]
        adj[(e + 1) % n, e] = weights[e]
    L_graph, _, _ = BuildGraphLaplace(AT.get_tensor(adj)).build()

    assert np.allclose(_np(L_dec), _np(L_graph))


# --------------------------------------------------------------------------
# leg 2: d . d == 0
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 7])
def test_boundary_of_boundary_vanishes(n: int) -> None:
    """``d1 @ d0 == 0`` -- the defining DEC identity, on the torus complex."""
    d0_np, d1_np = _torus_d0_d1(n)
    d0 = AT.get_tensor(d0_np)
    d1 = AT.get_tensor(d1_np)

    composed = d1 @ d0
    norm = float(AT.linalg.norm(composed))
    assert norm < 1e-9, f"DEC violation: ||d1 @ d0|| = {norm:.3e}"


# --------------------------------------------------------------------------
# leg 3: FFT diagonalises the circulant Laplacian
# --------------------------------------------------------------------------

def _circulant_spectrum_via_fft(L) -> "AT":
    """Eigenvalues of a circulant matrix: the DFT of its first row.

    Runs through the dispatched ``AbstractTensor.fft`` op, so this is the first
    path in the tree that routes a Laplacian through the spectral operator
    instead of a dense eigensolve.
    """
    return AT.real(L[0].fft())


@pytest.mark.parametrize("n", [8, 16])
def test_ring_spectrum_via_fft_matches_analytic(n: int) -> None:
    """FFT of the Laplacian's first row == ``2 - 2cos(2*pi*m/n)`` exactly."""
    L, _, _ = BuildGraphLaplace(AT.get_tensor(_ring_adjacency(n))).build()

    spectrum = _np(_circulant_spectrum_via_fft(L))
    analytic = np.array([2.0 - 2.0 * np.cos(2 * np.pi * m / n) for m in range(n)])

    assert np.allclose(spectrum, analytic, atol=1e-10), (
        f"fft spectrum {spectrum}\nanalytic     {analytic}"
    )
    # Equivalently 4 sin^2(pi m / n): the discrete symbol of -d^2/dx^2.
    assert np.allclose(
        spectrum, 4.0 * np.sin(np.pi * np.arange(n) / n) ** 2, atol=1e-10
    )


def test_ring_spectrum_via_fft_matches_eigh() -> None:
    """The FFT route and the dense Jacobi eigensolver agree.

    Regression guard for the ``sign(0) == 0`` collapse in ``eigh``: a ring
    Laplacian has a constant diagonal, so ``tau`` is identically zero and the
    Jacobi rotation used to vanish, returning the bare diagonal.
    """
    n = 8
    L, _, _ = BuildGraphLaplace(AT.get_tensor(_ring_adjacency(n))).build()

    fft_spectrum = np.sort(_np(_circulant_spectrum_via_fft(L)))
    eigh_values, _ = AT.eigh(L)
    eigh_spectrum = np.sort(_np(eigh_values))

    assert np.allclose(fft_spectrum, eigh_spectrum, atol=1e-8), (
        f"fft  {fft_spectrum}\neigh {eigh_spectrum}"
    )
    # The eigensolver must not simply be handing back the diagonal.
    assert not np.allclose(eigh_spectrum, np.diag(_np(L)))


def test_torus_spectrum_via_separable_fft() -> None:
    """2D spectrum by composing the 1-D dispatched FFT along each axis.

    ``AbstractTensor`` exposes only a 1-D ``fft(axis=...)``; a block-circulant
    (torus) Laplacian needs a 2-D transform, which is recovered by transforming
    each axis in turn.
    """
    n = 8
    L, _, _ = BuildGraphLaplace(AT.get_tensor(_torus_adjacency(n))).build()

    # Row 0 of L, reshaped to the (row, col) stencil it represents.
    stencil = L[0].reshape(n, n)
    spectrum = AT.real(stencil.fft(axis=0).fft(axis=1))

    a = np.arange(n)[:, None]
    b = np.arange(n)[None, :]
    analytic = (
        4.0 - 2.0 * np.cos(2 * np.pi * a / n) - 2.0 * np.cos(2 * np.pi * b / n)
    )

    assert np.allclose(_np(spectrum), analytic, atol=1e-10)


def test_fft_spectrum_survives_ifft_roundtrip() -> None:
    """``ifft(fft(row)) == row``, keeping the spectral route on the tape."""
    n = 16
    L, _, _ = BuildGraphLaplace(AT.get_tensor(_ring_adjacency(n))).build()
    row = L[0]
    restored = AT.real(row.fft().ifft())
    assert np.allclose(_np(restored), _np(row), atol=1e-10)


# --------------------------------------------------------------------------
# leg 4: the discrete symbol converges to the continuous Beltrami spectrum
# --------------------------------------------------------------------------

def _flat_metric(intrinsic_dim: int):
    def metric_function(query):
        samples = AT.get_tensor(query).get_shape()[0]
        return AT.get_tensor(np.eye(intrinsic_dim)[None, :, :].repeat(samples, 0))
    return metric_function


def _beltrami_eigenvalue(mode: int, samples: np.ndarray, *, step: float = 1e-4) -> float:
    """Measure the continuous ``-Delta`` eigenvalue of ``sin(mode * x)``.

    Uses the real ``continuous_laplace_beltrami`` operator with a flat metric,
    so this is the genuine metric operator evaluated on a Euclidean chart, not
    a hand-written second difference.
    """
    coords = samples.reshape(-1, 1)

    def gradient_function(query):
        q = _np(query)
        return AT.get_tensor(mode * np.cos(mode * q))

    laplacian = continuous_laplace_beltrami(
        AT.get_tensor(coords), _flat_metric(1), gradient_function, step=step
    )
    field = np.sin(mode * coords).ravel()
    # -Delta f = lambda f, evaluated where f is safely away from its zeros.
    return float(np.mean(-_np(laplacian) / field))


@pytest.mark.parametrize("mode", [1, 3])
def test_beltrami_recovers_plane_wave_eigenvalue(mode: int) -> None:
    """The continuous operator returns ``mode**2`` on ``sin(mode * x)``."""
    samples = np.array([0.3, 0.7, 1.1, 1.9])
    measured = _beltrami_eigenvalue(mode, samples)
    assert measured == pytest.approx(mode**2, rel=1e-4), (
        f"Beltrami eigenvalue {measured} != {mode**2}"
    )


@pytest.mark.parametrize("mode", [1, 3])
def test_discrete_symbol_converges_to_continuous_second_order(mode: int) -> None:
    """``fft-symbol / h^2 -> mode**2`` at second order as the ring refines.

    This is the seam that closes the set.  The ring of ``n`` nodes discretises a
    circle of circumference ``2*pi`` with spacing ``h = 2*pi/n``; the graph
    Laplacian approximates ``-h^2 * d^2/dx^2``, so its FFT symbol divided by
    ``h^2`` must approach the Beltrami eigenvalue, with error ``O(h^2)``.
    """
    continuous = _beltrami_eigenvalue(mode, np.array([0.3, 0.7, 1.1, 1.9]))

    errors = []
    for n in (32, 64, 128):
        h = 2 * np.pi / n
        L, _, _ = BuildGraphLaplace(AT.get_tensor(_ring_adjacency(n))).build()
        symbol = _np(_circulant_spectrum_via_fft(L))[mode] / (h * h)
        errors.append(abs(symbol - continuous))

    # Monotone convergence, and each halving of h must quarter the error.
    assert errors[0] > errors[1] > errors[2], f"not converging: {errors}"
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.05), (
            f"expected second-order convergence, got ratio {coarse / fine:.3f} "
            f"from errors {errors}"
        )
