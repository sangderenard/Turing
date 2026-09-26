"""Singularity stencils: classify a point by its punctured neighbourhood.

An ordinary stencil estimates the value at a point from its neighbours.  A
singularity stencil never evaluates the centre at all.  It walks a closed
contour around the centre and measures a discrete invariant of the field on
that contour -- the winding number (topological degree) of a planar vector
field.  The invariant stays exact while the contour avoids zeros of the
field, so it can report "two zeros live in here" even when the centre is NaN
and no sampled grid point comes near either zero.

The module is organised as an "iceberg" of increasingly expensive tiers:

* tier 0 -- :func:`ring_magnitude` : cheap magnitude evidence on the contour.
* tier 1 -- :func:`circle_stencil` / :func:`box_stencil` : winding number
  plus a self-check that the contour was sampled finely enough.
* tier 2 -- :func:`decompose` : recursive subdivision that splits a coarse
  charge into its constituent zeros (e.g. ``2 -> 1 + 1``).

The reference experiment uses ``F_a(z) = z**2 - a`` whose zeros sit at
``+-sqrt(a)``.  Topology predicts ``W(r, a) = 0`` for ``r < sqrt|a|`` and
``W = 2`` for ``r > sqrt|a|``; :func:`phase_diagram` measures it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

Field = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Reference fields
# ---------------------------------------------------------------------------


def z2_minus_a(a: float | torch.Tensor) -> Field:
    """Return the planar field of ``F(z) = z**2 - a``.

    ``a`` may be a tensor broadcastable against the sample coordinates, which
    lets :func:`phase_diagram` evaluate a whole sweep in one batched call.
    """

    def f(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x * x - y * y - a, 2.0 * x * y

    return f


def with_hole(f: Field, center: Tuple[float, float], radius: float) -> Field:
    """Wrap ``f`` so it returns NaN inside a disc: the centre is unavailable."""

    cx, cy = center

    def g(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fx, fy = f(x, y)
        hole = (x - cx) ** 2 + (y - cy) ** 2 < radius * radius
        nan = torch.full_like(fx, float("nan"))
        return torch.where(hole, nan, fx), torch.where(hole, nan, fy)

    return g


def smooth_noise(
    sigma: float,
    n_modes: int = 12,
    max_wavenumber: float = 3.0,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
) -> Field:
    """Random smooth vector field: a sum of random Fourier modes.

    The noise must be continuous -- pointwise i.i.d. noise has no topology --
    so it is built from ``n_modes`` plane waves per component, normalised so
    each component has RMS amplitude ``sigma``.
    """

    def draw(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, dtype=dtype)

    k = (draw(2, n_modes, 2) * 2.0 - 1.0) * max_wavenumber
    phase = draw(2, n_modes) * TWO_PI
    amp = sigma * math.sqrt(2.0 / n_modes)

    def eta(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        comps = []
        for c in range(2):
            arg = x.unsqueeze(-1) * k[c, :, 0] + y.unsqueeze(-1) * k[c, :, 1] + phase[c]
            comps.append(amp * torch.cos(arg).sum(-1))
        return comps[0], comps[1]

    return eta


def add_fields(f: Field, g: Field) -> Field:
    def h(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fx, fy = f(x, y)
        gx, gy = g(x, y)
        return fx + gx, fy + gy

    return h


# ---------------------------------------------------------------------------
# Tier 0/1: contour winding
# ---------------------------------------------------------------------------


def wrap_angle(d: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences into ``[-pi, pi)``."""

    return torch.remainder(d + math.pi, TWO_PI) - math.pi


def contour_winding(fx: torch.Tensor, fy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Winding of field samples ordered around a closed contour (last dim).

    Returns ``(winding, max_step)`` where ``max_step`` is the largest wrapped
    angular jump between consecutive samples.  If it approaches ``pi`` the
    contour is under-sampled (or passes near a zero) and the integer is not
    trustworthy.
    """

    phi = torch.atan2(fy, fx)
    d = wrap_angle(torch.roll(phi, -1, dims=-1) - phi)
    return d.sum(-1) / TWO_PI, d.abs().amax(-1)


def ring_magnitude(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Tier 0: smallest field magnitude seen on the contour."""

    return torch.sqrt(fx * fx + fy * fy).amin(-1)


@dataclass
class StencilReport:
    winding: int
    raw_winding: float
    min_magnitude: float
    max_step: float
    reliable: bool

    @property
    def is_singular(self) -> bool:
        return self.reliable and self.winding != 0


# Largest per-sample phase jump we accept before calling the integer unreliable.
RELIABLE_STEP = math.pi / 2


def _report(fx: torch.Tensor, fy: torch.Tensor) -> StencilReport:
    raw, step = contour_winding(fx, fy)
    mag = ring_magnitude(fx, fy)
    raw_f, step_f, mag_f = float(raw), float(step), float(mag)
    finite = math.isfinite(raw_f) and math.isfinite(mag_f)
    w = int(round(raw_f)) if finite else 0
    reliable = finite and step_f < RELIABLE_STEP and abs(raw_f - w) < 1e-6
    return StencilReport(w, raw_f, mag_f, step_f, reliable)


def circle_points(
    center: Tuple[float, float], radius: float | torch.Tensor, n: int, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = torch.arange(n, dtype=dtype) * (TWO_PI / n)
    r = torch.as_tensor(radius, dtype=dtype).unsqueeze(-1)
    return center[0] + r * torch.cos(theta), center[1] + r * torch.sin(theta)


def box_points(
    x0: float, y0: float, x1: float, y1: float, n_per_edge: int, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Counter-clockwise samples on the boundary of an axis-aligned box."""

    t = torch.arange(n_per_edge, dtype=dtype) / n_per_edge
    xs = torch.cat([x0 + (x1 - x0) * t, torch.full_like(t, x1), x1 - (x1 - x0) * t, torch.full_like(t, x0)])
    ys = torch.cat([torch.full_like(t, y0), y0 + (y1 - y0) * t, torch.full_like(t, y1), y1 - (y1 - y0) * t])
    return xs, ys


def circle_stencil(f: Field, center: Tuple[float, float], radius: float, n: int = 256) -> StencilReport:
    """Winding of ``f`` on a circle; the centre itself is never sampled."""

    x, y = circle_points(center, radius, n)
    return _report(*f(x, y))


def box_stencil(f: Field, box: Tuple[float, float, float, float], n_per_edge: int = 64) -> StencilReport:
    x, y = box_points(*box, n_per_edge)
    return _report(*f(x, y))


def phase_diagram(
    a_values: torch.Tensor, r_values: torch.Tensor, n: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure ``W(r, a)`` for ``F_a = z**2 - a`` on origin-centred circles.

    Returns ``(winding, min_magnitude, max_step)`` each shaped ``(len(r), len(a))``.
    Every circle is sampled with the centre removed.
    """

    a = a_values.to(torch.float64).view(1, -1, 1)
    r = r_values.to(torch.float64).view(-1)
    x, y = circle_points((0.0, 0.0), r, n)  # (R, n)
    x, y = x.unsqueeze(1), y.unsqueeze(1)  # (R, 1, n)
    fx, fy = z2_minus_a(a)(x, y)
    w, step = contour_winding(fx, fy)
    return w, ring_magnitude(fx, fy), step


def predicted_winding(a_values: torch.Tensor, r_values: torch.Tensor) -> torch.Tensor:
    """Analytic prediction: 2 if the circle encloses both zeros, else 0."""

    a = a_values.to(torch.float64).view(1, -1)
    r = r_values.to(torch.float64).view(-1, 1)
    return torch.where(r > torch.sqrt(a.abs()), 2.0, 0.0)


# ---------------------------------------------------------------------------
# Baseline: naive pointwise zero detection
# ---------------------------------------------------------------------------


def naive_grid_zeros(f: Field, extent: float, spacing: float, eps: float) -> torch.Tensor:
    """Grid points where ``|F| < eps`` -- what a pointwise detector sees."""

    n = int(round(2 * extent / spacing)) + 1
    g = torch.linspace(-extent, extent, n, dtype=torch.float64)
    x, y = torch.meshgrid(g, g, indexing="xy")
    fx, fy = f(x, y)
    mag = torch.sqrt(fx * fx + fy * fy)
    hit = torch.nan_to_num(mag, nan=float("inf")) < eps
    return torch.stack([x[hit], y[hit]], dim=-1)


# ---------------------------------------------------------------------------
# Tier 2: decomposition of a coarse charge
# ---------------------------------------------------------------------------


@dataclass
class Charge:
    box: Tuple[float, float, float, float]
    winding: int

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    @property
    def size(self) -> float:
        return self.box[2] - self.box[0]


@dataclass
class Decomposition:
    total: int
    charges: List[Charge] = field(default_factory=list)
    unresolved: List[Tuple[float, float, float, float]] = field(default_factory=list)


def decompose(
    f: Field,
    box: Tuple[float, float, float, float],
    min_size: float = 1e-3,
    n_per_edge: int = 64,
    max_depth: int = 32,
) -> Decomposition:
    """Split the charge enclosed by ``box`` into localised pieces.

    Box contours are additive: the four children of a box tile it, so their
    windings sum to the parent's.  Children with ``W = 0`` are discarded.  A
    box stops splitting once it is smaller than ``min_size``; every surviving
    charge is therefore pinned to within ``min_size`` of a zero (or of a
    cluster of zeros that has not separated at that scale).  Boxes whose own
    contour is unreliable -- a zero sitting on an edge -- are nudged by
    splitting off-centre, and recorded as ``unresolved`` only if that fails.
    """

    root = box_stencil(f, box, n_per_edge)
    out = Decomposition(total=root.winding if root.reliable else 0)
    if not root.reliable:
        out.unresolved.append(box)
        return out

    def recurse(b: Tuple[float, float, float, float], w: int, depth: int) -> None:
        x0, y0, x1, y1 = b
        if w == 0:
            return
        if (x1 - x0) <= min_size or depth >= max_depth:
            out.charges.append(Charge(b, w))
            return
        # An off-centre split avoids pathological symmetric placements where a
        # zero lies exactly on the midline (as +-sqrt(a) do for y = 0 splits
        # through the real axis combined with x = 0 splits between them).
        for frac in (0.5 + 1e-3, 0.5 - 7e-3, 0.5 + 3.1e-2):
            xm = x0 + (x1 - x0) * frac
            ym = y0 + (y1 - y0) * frac
            kids = [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]
            reps = [box_stencil(f, k, n_per_edge) for k in kids]
            if all(r.reliable for r in reps) and sum(r.winding for r in reps) == w:
                for k, r in zip(kids, reps):
                    recurse(k, r.winding, depth + 1)
                return
        out.unresolved.append(b)

    recurse(box, root.winding, 0)
    return out


def detection_rate(
    a: float,
    radius: float,
    sigmas: Sequence[float],
    trials: int = 64,
    n: int = 256,
    seed: int = 0,
) -> List[float]:
    """Fraction of noisy trials in which the circle stencil still reports 2."""

    gen = torch.Generator().manual_seed(seed)
    rates = []
    for sigma in sigmas:
        hits = 0
        for _ in range(trials):
            f = add_fields(z2_minus_a(a), smooth_noise(sigma, generator=gen))
            rep = circle_stencil(f, (0.0, 0.0), radius, n)
            hits += int(rep.reliable and rep.winding == 2)
        rates.append(hits / trials)
    return rates
