"""A driven resonant cavity on a DEC complex, against what a cavity must do.

Everything here was first measured with throwaway probes; this module
exists so those measurements cannot quietly stop being true.  The checks
are the same kind used for the DEC core itself -- an analytic answer, a
convergence RATE, and agreement between two independent routes to the
same field -- rather than tolerances chosen to fit whatever came out.

The one bug this is most meant to catch: the modal solver's loss term
had its sign flipped.  Off resonance that barely shows, because the real
part of the denominator dominates.  ON resonance the real part vanishes,
the answer is exactly out of phase, and modal-versus-direct measured
199.8% against a direct solve whose field the modes held to 99.88%.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.abstract_convolution.dec_system import DECSystem
from src.common.tensors.abstract_convolution.dec_cavity import (
    CavityModes,
    DrivenCavity,
    SPEED_OF_LIGHT,
    feed_at,
    perfect_conductor_mask,
)


SIZE = (0.30, 0.30, 0.20)
#: Shapes whose spacing is isotropic for SIZE: h = 0.3/(n-1) = 0.2/(m-1).
SHAPES = [(7, 7, 5), (10, 10, 7), (13, 13, 9)]


def _cavity(shape, size=SIZE):
    spacing = size[0] / (shape[0] - 1)
    system = DECSystem.from_lattice(shape, spacing=spacing)
    return system, DrivenCavity(system), spacing


def _analytic_modes(size=SIZE, limit=8):
    """``f_mnp = (c/2) sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``, at most one zero."""
    a, b, d = size
    found = [
        SPEED_OF_LIGHT / 2 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
        for m in range(4) for n in range(4) for p in range(4)
        if (m == 0) + (n == 0) + (p == 0) <= 1
    ]
    return np.sort(np.unique(np.round(found, 3)))[:limit]


def _face_loop(system, face: int) -> np.ndarray:
    """One face's own boundary chain: a current loop, solenoidal by ``d.d = 0``."""
    row = system.sparse("d1")[face]
    loop = np.zeros(system.complex.edge_count, dtype=np.complex128)
    loop[row.indices] = row.data
    return loop


# --------------------------------------------------------------------------
# the walls
# --------------------------------------------------------------------------

def test_a_perfect_conductor_removes_the_edges_lying_in_its_walls() -> None:
    """``n x E = 0`` is exactly "these edges carry nothing".

    The generic route -- edges of faces that bound only one cell -- must
    agree with the box-specific one, which is that both endpoints share a
    wall plane.
    """
    shape = (7, 7, 5)
    system, _, _ = _cavity(shape)
    free = perfect_conductor_mask(system)

    grid = np.array([[i, j, k] for k in range(shape[2])
                     for j in range(shape[1]) for i in range(shape[0])])
    tail, head = system.complex.edges[:, 0], system.complex.edges[:, 1]
    on_a_wall = np.zeros(system.complex.edge_count, dtype=bool)
    for axis, extent in enumerate(shape):
        for wall in (0, extent - 1):
            on_a_wall |= ((grid[tail][:, axis] == wall)
                          & (grid[head][:, axis] == wall))

    assert np.array_equal(~free, on_a_wall)
    assert free.sum() == 280


# --------------------------------------------------------------------------
# the resonances
# --------------------------------------------------------------------------

def test_cavity_modes_match_the_analytic_rectangular_formula() -> None:
    """The lowest modes, and their degeneracies, on the finest grid.

    Degeneracy is structural: a square cross-section makes the second and
    third modes the same frequency, and an operator that broke the
    symmetry would split them.
    """
    system, cavity, _ = _cavity((13, 13, 9))
    modes = CavityModes(cavity, count=8, resolve_hz=9e8)

    found = modes.frequencies_hz
    analytic = _analytic_modes(limit=1)
    assert abs(found[0] - analytic[0]) / analytic[0] < 0.004
    # Second and third are one degenerate pair, to the solver's precision.
    assert abs(found[1] - found[2]) / found[1] < 1e-6
    assert abs(found[3] - found[4]) / found[3] < 1e-6


@pytest.mark.parametrize("shape", SHAPES)
def test_every_resolution_lands_near_the_analytic_fundamental(shape) -> None:
    system, cavity, _ = _cavity(shape)
    # Resolve NEAR the fundamental: shift-invert returns the modes closest
    # to the shift, so asking around 900 MHz for four modes never reaches
    # down to 706 MHz and the comparison would be against the wrong mode.
    modes = CavityModes(cavity, count=4, resolve_hz=7.1e8)

    analytic = _analytic_modes(limit=1)[0]
    assert abs(modes.frequencies_hz[0] - analytic) / analytic < 0.02


def test_the_fundamental_converges_at_second_order() -> None:
    """Halve the spacing, quarter the error -- a RATE, not a tolerance.

    This is the strongest statement available about a discretisation, and
    it is what separates "close enough on this grid" from "correct".
    Measured 1.138%, 0.507%, 0.285% at h = 50, 33.3, 25 mm.
    """
    analytic = _analytic_modes(limit=1)[0]
    errors, spacings = [], []
    for shape in SHAPES:
        system, cavity, spacing = _cavity(shape)
        modes = CavityModes(cavity, count=4, resolve_hz=7.1e8)
        errors.append(abs(modes.frequencies_hz[0] - analytic) / analytic)
        spacings.append(spacing)

    for before, after in zip(range(len(SHAPES) - 1), range(1, len(SHAPES))):
        ratio = errors[after] / errors[before]
        expected = (spacings[after] / spacings[before]) ** 2
        assert abs(ratio - expected) / expected < 0.15, (
            f"error ratio {ratio:.4f} is not second order ({expected:.4f})")


# --------------------------------------------------------------------------
# asking for more modes than exist
# --------------------------------------------------------------------------

def test_more_modes_than_the_cavity_has_returns_its_gradients() -> None:
    """Weyl's law bounds how many modes are there; the rest is ``im(d0)``.

    Shift-invert ranks by ``|1/(lambda - sigma)|``, so the null space at
    zero scores ``1/sigma`` -- small, taken last, but taken once the
    genuine modes run out.  Asking for thirty near 900 MHz returned
    twenty-one gradients before this was filtered, and they arrive as
    frequencies of a few tens of HERTZ, which is easy to misread as a
    scale error rather than as nothing at all.
    """
    system, cavity, _ = _cavity((13, 13, 9))

    modest = CavityModes(cavity, count=8, resolve_hz=9e8)
    greedy = CavityModes(cavity, count=40, resolve_hz=9e8)

    assert greedy.discarded_gradients > 0
    assert greedy.mode_count < 40
    # The genuine modes are the same ones either way.
    shared = min(modest.mode_count, greedy.mode_count)
    assert np.allclose(modest.frequencies_hz[:shared],
                       greedy.frequencies_hz[:shared], rtol=1e-6)
    # And none of the survivors is a gradient.
    assert (greedy.frequencies_hz > 1e6).all()


# --------------------------------------------------------------------------
# modal against direct
# --------------------------------------------------------------------------

def test_a_face_boundary_is_a_solenoidal_current_loop() -> None:
    """``d0^T *1 j == 0`` exactly, because ``d1 d0 == 0``.

    A non-solenoidal feed drives a gradient response that no curl mode
    can hold, so the loop is what makes modal-versus-direct a fair test
    of the coefficients rather than of completeness.
    """
    system, cavity, _ = _cavity((9, 9, 7))
    loop = _face_loop(system, system.complex.face_count // 2 + 7)

    d0 = system.sparse("d0").tocsc()[cavity.index, :]
    divergence = d0.T @ (cavity.star1 * loop[cavity.index])

    assert np.linalg.norm(divergence) < 1e-12


def test_modal_and_direct_agree_on_resonance() -> None:
    """Two independent routes to the same field, where it matters most.

    On resonance the modes hold essentially all of the direct field, so
    what remains is whether the coefficients are right.  This is the
    test that fails loudly -- around 200% -- if the loss term's sign is
    flipped back.
    """
    system, cavity, _ = _cavity((13, 13, 9))
    modes = CavityModes(cavity, count=10, resolve_hz=9e8)
    loop = _face_loop(system, system.complex.face_count // 2 + 7)

    for index in (1, 3):
        frequency = modes.frequencies_hz[index]
        direct = cavity.solve(frequency, loop, quality=600.0).field
        modal = modes.solve(frequency, loop, quality=600.0).field

        held = (np.linalg.norm(modes.vectors @ (modes.vectors.T
                                                @ (cavity.star1 * direct)))
                / np.linalg.norm(direct))
        assert held > 0.99, f"modes hold only {held:.2%} of the direct field"
        error = np.linalg.norm(direct - modal) / np.linalg.norm(direct)
        assert error < 0.10, f"modal differs from direct by {error:.2%}"


def test_the_response_peaks_at_the_cavity_modes() -> None:
    """A sweep is what a cavity is FOR, and its peaks are its modes."""
    system, cavity, spacing = _cavity((13, 13, 9))
    modes = CavityModes(cavity, count=6, resolve_hz=9e8)
    feed = feed_at(system, (0.075, 0.125, 0.10), axis=2, width=1.2 * spacing)

    frequencies = np.linspace(650e6, 1150e6, 501)
    amplitude = modes.sweep(frequencies, feed, quality=600.0)

    peaks = [i for i in range(1, len(amplitude) - 1)
             if amplitude[i] > amplitude[i - 1]
             and amplitude[i] > amplitude[i + 1]
             and amplitude[i] > 0.02 * amplitude.max()]
    assert peaks, "a driven cavity with modes in band must resonate"
    for peak in peaks:
        nearest = np.min(np.abs(modes.frequencies_hz - frequencies[peak]))
        assert nearest / frequencies[peak] < 0.01


def test_a_sweep_is_far_cheaper_through_the_modes_than_directly() -> None:
    """The reason the modes exist: one eigensolve, then arithmetic.

    A direct solve measured about 1.2 s at 2816 free edges because a
    three-dimensional curl-curl factorisation fills in badly.  The same
    sweep through the modes is a projection per frequency.
    """
    import time

    system, cavity, spacing = _cavity((13, 13, 9))
    modes = CavityModes(cavity, count=6, resolve_hz=9e8)
    feed = feed_at(system, (0.075, 0.125, 0.10), axis=2, width=1.2 * spacing)

    start = time.perf_counter()
    modes.sweep(np.linspace(650e6, 1150e6, 201), feed, quality=600.0)
    modal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    cavity.solve(9.0e8, feed, quality=600.0)
    direct_seconds = time.perf_counter() - start

    assert modal_seconds < direct_seconds, (
        f"201 modal solves took {modal_seconds:.3f}s against one direct "
        f"solve at {direct_seconds:.3f}s")
