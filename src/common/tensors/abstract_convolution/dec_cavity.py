"""A resonant cavity driven at one frequency, solved on a DEC complex.

The time-harmonic Maxwell problem for the electric field in a cavity is

    curl (1/mu) curl E  -  omega^2 eps E  =  -i omega J

and on a DEC complex that is, with no vector calculus left in it,

    ( d1^T *_(1/mu) d1  -  k^2 *_eps ) e  =  -i omega mu0 j,   k = omega / c

``e`` is the 1-form of edge line integrals and ``j`` the driving current
on the same edges.  A perfect conductor removes the edges lying in the
wall, because ``n x E = 0`` says exactly that their line integrals
vanish; :meth:`CellComplex.boundary_edges` names them.

WHY THIS IS SOLVED DIRECTLY RATHER THAN STEPPED.  Time-stepping a
microwave cavity is governed by ``dt <= h / (c sqrt(3))`` -- about ten
picoseconds for a five-millimetre cell -- and steady state is hundreds
of periods away, so nothing is worth looking at for ~1e5 steps.  The
driven steady state is one linear solve.  It is also not a modal sum:
an oven at 2.45 GHz has of order eighty modes beneath it, and the
eigenproblem is the expensive way to answer a question about a single
frequency.

LOSS.  A cavity with perfectly conducting walls has a singular response
exactly on resonance.  Real walls have finite conductivity, and the
standard way to carry that at one frequency is a complex permittivity --
``eps -> eps (1 + i/Q)`` -- which bounds the response and gives every
resonance a finite width.  ``quality`` is that Q.  It is a declared
property of the cavity, not a fitted number: it comes from the wall's
surface resistance, the same skin-depth arithmetic the shielding lane
uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMEABILITY = 4.0e-7 * np.pi


@dataclass(frozen=True)
class DrivenResponse:
    """One frequency's answer, on the edges that were free to move."""

    frequency_hz: float
    field: np.ndarray            # complex, one entry per FREE edge
    free_edges: np.ndarray
    stored_energy: float

    @property
    def amplitude(self) -> float:
        """A single number per frequency, for sweeps."""
        return float(np.linalg.norm(self.field))


def perfect_conductor_mask(system) -> np.ndarray:
    """Edges free to carry a field: everything not lying in a wall."""
    free = np.ones(system.complex.edge_count, dtype=bool)
    free[system.complex.boundary_edges()] = False
    return free


class DrivenCavity:
    """One cavity, assembled once, solvable at many frequencies.

    The curl-curl operator does not depend on the drive, so a sweep that
    rebuilds it per frequency pays for the assembly hundreds of times
    over.  Only the ``k^2 *1`` term moves, and it is diagonal.
    """

    def __init__(self, system, *, free: np.ndarray | None = None):
        from scipy.sparse import diags

        self.system = system
        self.free = perfect_conductor_mask(system) if free is None else free
        self.index = np.flatnonzero(self.free)
        d1 = system.sparse("d1")
        star2 = np.asarray(system.star2.data, dtype=np.float64)
        curl_curl = (d1.T @ diags(star2) @ d1).tocsr()
        self.curl_curl = curl_curl[self.index][:, self.index].astype(np.complex128)
        self.star1 = np.asarray(
            system.star1.data, dtype=np.float64)[self.index]

    def operator(self, frequency_hz: float, *, quality: float | None = None):
        from scipy.sparse import diags

        wavenumber = 2.0 * np.pi * float(frequency_hz) / SPEED_OF_LIGHT
        mass = self.star1.astype(np.complex128)
        if quality is not None and quality > 0:
            # Complex permittivity: the standard one-frequency loss spelling.
            mass = mass * (1.0 + 1.0j / float(quality))
        return self.curl_curl - (wavenumber ** 2) * diags(mass)

    def solve(self, frequency_hz: float, current: np.ndarray, *,
              quality: float | None = 200.0) -> "DrivenResponse":
        from scipy.sparse.linalg import spsolve

        source = np.asarray(current, dtype=np.complex128)
        if source.shape != (self.system.complex.edge_count,):
            raise ValueError(
                f"current needs one entry per edge "
                f"({self.system.complex.edge_count})")
        omega = 2.0 * np.pi * float(frequency_hz)
        rhs = -1.0j * omega * VACUUM_PERMEABILITY * source[self.index]
        field = spsolve(self.operator(frequency_hz, quality=quality), rhs)
        stored = float(np.sum(self.star1 * np.abs(field) ** 2))
        return DrivenResponse(float(frequency_hz), field, self.index, stored)

    def sweep(self, frequencies, current, *, quality: float | None = 200.0):
        """Amplitude at each frequency: the cavity's response curve."""
        return np.array([
            self.solve(frequency, current, quality=quality).amplitude
            for frequency in frequencies])


def drive(system, frequency_hz: float, current: np.ndarray, *,
          quality: float | None = 200.0,
          free: np.ndarray | None = None) -> DrivenResponse:
    """Solve the cavity once, for one frequency and one driving current.

    ``current`` is a real or complex 1-form over ALL edges, so a feed may
    be declared where it physically sits without knowing the boundary.
    For more than one frequency build a :class:`DrivenCavity` instead and
    reuse its assembly.
    """
    return DrivenCavity(system, free=free).solve(
        frequency_hz, current, quality=quality)


class CavityModes:
    """The cavity's own modes, solved once, reused at every frequency.

    A direct solve answers ONE frequency, and on a three-dimensional
    curl-curl operator it is expensive: measured 1.17 s for 2816 free
    edges, because the factorization fills in badly and the fill grows
    faster than the problem.  A sweep or an animation asks hundreds of
    frequencies, and for those the modes are the cheap route -- one
    shift-inverted eigensolve, after which each frequency is a sum:

        a_m = -i omega mu0 (e_m . j) / (lambda_m - k^2)
        e   = sum_m a_m e_m

    ``resolve`` is the shift the eigensolver works around; modes are
    returned nearest to it, which is where a drive in that band couples.
    The gradient null space sits at zero and is far from any useful
    shift, so it neither needs nor gets special handling.
    """

    #: A mode is genuine when its eigenvalue is this fraction of the shift
    #: or more.  The gradients sit at machine zero -- measured 3e-12
    #: against a shift of 356 -- so any threshold in between separates
    #: them cleanly and none of them survives.
    GRADIENT_TOLERANCE = 1e-6

    def __init__(self, cavity: "DrivenCavity", count: int = 12,
                 resolve_hz: float = 1.0e9):
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh

        self.cavity = cavity
        shift = (2.0 * np.pi * float(resolve_hz) / SPEED_OF_LIGHT) ** 2
        mass = diags(cavity.star1)
        values, vectors = eigsh(
            cavity.curl_curl.real.tocsc(), k=int(count), M=mass.tocsc(),
            sigma=shift, which="LM")

        # ASK FOR MORE MODES THAN THE CAVITY HAS AND YOU GET ITS GRADIENTS.
        # Shift-invert ranks by |1 / (lambda - sigma)|, and the null space
        # at lambda = 0 scores 1/sigma -- small, so it is taken last, but
        # taken nonetheless once the genuine modes run out.  Weyl's law
        # says how many exist: N(f) ~ 8 pi V f^3 / 3 c^3, which for a
        # 0.3 x 0.3 x 0.2 m box is about eleven below 1.27 GHz.  Asking
        # for thirty returned twenty-one gradients.
        genuine = values > self.GRADIENT_TOLERANCE * shift
        self.discarded_gradients = int((~genuine).sum())
        values, vectors = values[genuine], vectors[:, genuine]
        if not len(values):
            raise ValueError(
                f"no cavity mode found near {resolve_hz/1e9:.3f} GHz; "
                "every eigenvalue returned was a gradient")

        order = np.argsort(values)
        self.eigenvalues = values[order]
        self.vectors = vectors[:, order]
        # M-orthonormal, so the modal coefficients below are plain
        # projections rather than a solve.
        norms = np.sqrt(np.einsum(
            "im,i,im->m", self.vectors, cavity.star1, self.vectors))
        self.vectors = self.vectors / norms

    @property
    def mode_count(self) -> int:
        return len(self.eigenvalues)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return SPEED_OF_LIGHT * np.sqrt(np.maximum(self.eigenvalues, 0.0)) / (2 * np.pi)

    def solve(self, frequency_hz: float, current: np.ndarray, *,
              quality: float | None = 200.0) -> DrivenResponse:
        index = self.cavity.index
        source = np.asarray(current, dtype=np.complex128)[index]
        omega = 2.0 * np.pi * float(frequency_hz)
        wavenumber = (omega / SPEED_OF_LIGHT) ** 2
        # The mass matrix belongs to the ORTHONORMALITY, not to this
        # projection.  With v_m^T M v_n = delta_mn, projecting
        # (A - k^2 M) e = rhs onto v_n leaves a_n (lambda_n - k^2) =
        # v_n^T rhs, so weighting the source by *1 here double-counts it.
        projection = self.vectors.T @ source
        detuning = self.eigenvalues.astype(np.complex128) - wavenumber
        if quality is not None and quality > 0:
            # The direct operator carries loss as -k^2 *1 (1 + i/Q), so the
            # projected denominator is (lambda - k^2) - i k^2 / Q.  With
            # the sign flipped, off resonance it barely shows -- but ON
            # resonance the real part vanishes and the answer is exactly
            # out of phase, which measured as a 199.8% discrepancy against
            # a direct solve whose field the modes held to 99.88%.
            detuning = detuning - 1.0j * wavenumber / float(quality)
        coefficients = -1.0j * omega * VACUUM_PERMEABILITY * projection / detuning
        field = self.vectors.astype(np.complex128) @ coefficients
        stored = float(np.sum(self.cavity.star1 * np.abs(field) ** 2))
        return DrivenResponse(float(frequency_hz), field, index, stored)

    def sweep(self, frequencies, current, *, quality: float | None = 200.0):
        return np.array([
            self.solve(frequency, current, quality=quality).amplitude
            for frequency in frequencies])


def feed_at(system, position, *, axis: int = 2, width: float = 1.0):
    """A unit current on the edges nearest one point, along one axis.

    A real feed is a probe or an iris; this is the smallest honest stand
    in for one -- a short current element with a direction, which is what
    a probe is. ``position`` is in the same units the complex's vertices
    were built with.
    """
    edges = system.complex.edges
    vertices = getattr(system, "vertex_positions", None)
    if vertices is None:
        raise ValueError(
            "this complex carries no vertex positions; pass a current directly")
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    direction = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    along = np.abs(direction[:, axis]) > 0.5 * np.abs(direction).max()

    offset = midpoints - np.asarray(position, dtype=np.float64)
    near = np.linalg.norm(offset, axis=1) <= width
    current = np.zeros(system.complex.edge_count, dtype=np.complex128)
    chosen = np.flatnonzero(along & near)
    if not len(chosen):
        raise ValueError("no edge lies within the feed's width of that point")
    current[chosen] = 1.0 / len(chosen)
    return current
