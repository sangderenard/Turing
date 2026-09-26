"""Piecewise-cubic interpolants in AbstractTensor, with exact calculus.

Every interpolant here is a piecewise cubic on the data's knots, stored as
per-interval coefficients in the local variable ``s = x - x_i``::

    p_i(s) = c0 + c1 s + c2 s**2 + c3 s**3        on [x_i, x_{i+1}]

so evaluation, every derivative, the antiderivative and definite integrals
are exact closed forms, not finite differences or trapezoid sums.  The
methods differ only in how the knot slopes are chosen (the Hermite family)
or in the coefficients themselves (linear):

* ``linear``       straight segments (c2 = c3 = 0)
* ``natural_cubic``  C2 cubic spline with zero curvature at both ends
* ``pchip``        Fritsch-Carlson monotone slopes: never overshoots
                   monotone data (the slope form SciPy's PCHIP uses)
* ``akima``        Akima's locally weighted slopes: robust to outliers

Built on the antiderivative:

* ``bin_averages(edges)``  area-preserving resampling: the mean over each
  bin, ``(F(b) - F(a)) / (b - a)``, so the total is kept exactly
* ``window_maxima(width, starts)``  the best mean over any window of that
  width among the given starts (a maximal-mean curve)

Everything is vectorized AbstractTensor arithmetic: points are located by
comparison against the knots and coefficients are selected with a one-hot
contraction, so there is no Python-level loop over points or intervals.

``limbs=2`` (or more) makes every output CORRECTLY ROUNDED.  Three steps,
each measured necessary on a 69,120-output sweep (72 datasets: uniform,
random and clustered knots; smooth, noisy, step, offset, 12-decade dynamic
range and zero-crossing data; all methods; values, first derivatives,
antiderivatives, bin averages -- 0 misses against an exact Fraction oracle):

1. The coefficients are built EXACTLY from the data on the rational system
   (``RationalPrecision``: deferred quotients, one division per coefficient
   at readout) and captured at ``limbs`` limbs.  Float coefficients alone
   leave 3-600 ULP.
2. Every evaluation runs in ``Precision[limbs]``: the local coordinate is an
   exact two-sum and each coefficient is selected limb by limb (a 0/1
   contraction is exact), then collapsed once.
3. Each point is evaluated from its NEARER knot: every interval also carries
   the exact Taylor shift of its piece to the right knot.  Anchoring every
   point at the left knot leaves cancellation wells beside the right knot
   (an exact zero there came back as a 2**-106-sized residue: infinite ULP).
   Anchored at the nearer knot, a knot returns its data exactly and nothing
   beside it cancels.
"""

from __future__ import annotations

from typing import Any

from .abstraction import AbstractTensor
from . import linalg


def _tensor(value: Any) -> AbstractTensor:
    if isinstance(value, AbstractTensor):
        return value * 1.0
    return AbstractTensor.get_tensor(value) * 1.0


# --------------------------------------------------------------------------
# slope rules (each returns one slope per knot)
#
# Written only in operations every numerical type here supports -- element
# slicing, ``+ - * /``, ``sign``, ``abs``, comparisons, and the
# feature-aware ``where`` / ``concat`` / ``solve_tridiagonal`` of
# ``extended_precision`` -- so ONE implementation serves the float path
# (AbstractTensor) and the exact build (RationalPrecision).


def _secants(x, y):
    h = x[1:] - x[:-1]
    return h, (y[1:] - y[:-1]) / h


def natural_cubic_slopes(x, y):
    """Slopes of the natural cubic spline (second derivative zero at both ends).

    Solves the standard tridiagonal system for the knot second derivatives
    ``M``: ``h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1} =
    6 (d_i - d_{i-1})`` with ``M_0 = M_{K-1} = 0``, then reads each slope
    from its interval: ``m_i = d_i - h_i (2 M_i + M_{i+1}) / 6``.
    """

    from .extended_precision import concat, solve_tridiagonal

    h, d = _secants(x, y)
    zero = h[:1] * 0.0
    one = zero + 1.0
    lower = concat([zero, h[:-1], zero], dim=0)
    diagonal = concat([one, (h[:-1] + h[1:]) * 2.0, one], dim=0)
    upper = concat([zero, h[1:], zero], dim=0)
    rhs = concat([zero, (d[1:] - d[:-1]) * 6.0, zero], dim=0)
    second = solve_tridiagonal(lower, diagonal, upper, rhs)
    left_slopes = d - h * (second[:-1] * 2.0 + second[1:]) / 6.0          # knots 0..K-2
    last = d[-1:] + h[-1:] * (second[-2:-1] + second[-1:] * 2.0) / 6.0
    return concat([left_slopes, last], dim=0)


def pchip_slopes(x, y):
    """Fritsch-Carlson monotone slopes.

    Interior knot: the weighted harmonic mean of the neighbouring secants
    when they have the same sign, zero otherwise (a local extremum is kept
    flat so the curve cannot overshoot).  End knots: the one-sided
    three-point estimate, clipped to zero when its sign disagrees with the
    first secant and to three times the secant when the secants change sign.
    """

    from .extended_precision import concat, where

    h, d = _secants(x, y)
    h0, h1 = h[:-1], h[1:]
    d0, d1 = d[:-1], d[1:]
    w1 = h1 * 2.0 + h0
    w2 = h1 + h0 * 2.0
    same = (d0 * d1) > 0.0
    safe0 = where(same, d0, d0 * 0.0 + 1.0)
    safe1 = where(same, d1, d1 * 0.0 + 1.0)
    interior = where(same, (w1 + w2) / (w1 / safe0 + w2 / safe1), d0 * 0.0)

    def edge(ha, hb, da, db):
        estimate = ((ha * 2.0 + hb) * da - ha * db) / (ha + hb)
        wrong_sign = (estimate.sign() != da.sign())
        overshoot = (da.sign() != db.sign()) * (abs(estimate) > abs(da) * 3.0)
        estimate = where(overshoot, da * 3.0, estimate)
        return where(wrong_sign, estimate * 0.0, estimate)

    first = edge(h[:1], h[1:2], d[:1], d[1:2])
    last = edge(h[-1:], h[-2:-1], d[-1:], d[-2:-1])
    return concat([first, interior, last], dim=0)


def akima_slopes(x, y):
    """Akima's slopes: each knot's slope weights its neighbouring secants by
    how far the secants on the OTHER side disagree.

    ``m_i = (|d_{i+1} - d_i| d_{i-1} + |d_{i-1} - d_{i-2}| d_i) / (|d_{i+1} - d_i| + |d_{i-1} - d_{i-2}|)``
    with the secant sequence extended by two linear extrapolations at each
    end, and the plain average ``(d_{i-1} + d_i) / 2`` where both weights
    vanish (locally straight data).
    """

    from .extended_precision import concat, where

    _h, d = _secants(x, y)
    before1 = d[:1] * 2.0 - d[1:2]
    before2 = before1 * 2.0 - d[:1]
    after1 = d[-1:] * 2.0 - d[-2:-1]
    after2 = after1 * 2.0 - d[-1:]
    ext = concat([before2, before1, d, after1, after2], dim=0)            # K + 3
    count = int(x.shape[0])
    dm2 = ext[0:count]
    dm1 = ext[1:count + 1]
    d0 = ext[2:count + 2]
    dp1 = ext[3:count + 3]
    w_left = abs(dp1 - d0)
    w_right = abs(dm1 - dm2)
    total = w_left + w_right
    flat = total.sign() == 0
    safe_total = where(flat, total * 0.0 + 1.0, total)
    weighted = (w_left * dm1 + w_right * d0) / safe_total
    return where(flat, (dm1 + d0) * 0.5, weighted)


_SLOPE_RULES = {
    "natural_cubic": natural_cubic_slopes,
    "pchip": pchip_slopes,
    "akima": akima_slopes,
}


def _locate_nearer_columns(x, limbs: int, points: AbstractTensor):
    """Locate points using only the post-build tensor-column surface."""

    from .extended_precision import Precision

    flat = points.reshape(-1)
    count = int(x.shape[0])
    inner = x[1:-1]
    index = (flat.reshape(-1, 1) >= inner.reshape(1, -1)).sum(1) * 1.0
    intervals = AbstractTensor.arange(count - 1) * 1.0
    one_hot = (index.reshape(-1, 1) == intervals.reshape(1, -1)) * 1.0
    left = (one_hot * x[:-1].reshape(1, -1)).sum(1)
    right_knot = (one_hot * x[1:].reshape(1, -1)).sum(1)
    right = ((flat - left) > (right_knot - flat)) * 1.0
    anchor = left * (1.0 - right) + right_knot * right
    offset = Precision.of(flat, limbs) - Precision.of(anchor, limbs)
    return one_hot, right, offset


def _pick_columns(one_hot, right, left_limbs, right_limbs, limbs: int):
    """Select one post-build coefficient, preserving every authored limb."""

    from .extended_precision import Precision

    keep = 1.0 - right
    def selected(index):
        left = left_limbs[index]
        right_column = right_limbs[index]
        return (
            (one_hot * left.reshape(1, -1)).sum(1) * keep
            + (one_hot * right_column.reshape(1, -1)).sum(1) * right
        )

    result = Precision.of(selected(0), limbs)
    for index in range(1, limbs):
        result = result + Precision.of(selected(index), limbs)
    return result


def _wide_from_columns(
    x,
    limbs: int,
    points: AbstractTensor,
    what: str,
    order: int,
    left_columns,
    right_columns,
    F_left,
    F_right,
):
    """Evaluate the exact coefficients after ``limb_terms`` readout.

    This is the compiler-facing numerical boundary of the interpolant.  The
    rational build remains the eager producer of these ordinary tensor
    columns; the evaluator consumes no rational wrapper at its ABI.
    """

    one_hot, right, s = _locate_nearer_columns(x, limbs, points)
    # A cubic owns exactly four coefficients.  Keep those four identities
    # explicit so callsite specialization sees each coefficient's complete
    # limb tuple; ``limbs`` itself remains an arbitrary specialized width.
    c0 = _pick_columns(
        one_hot, right, left_columns[0], right_columns[0], limbs,
    )
    c1 = _pick_columns(
        one_hot, right, left_columns[1], right_columns[1], limbs,
    )
    c2 = _pick_columns(
        one_hot, right, left_columns[2], right_columns[2], limbs,
    )
    c3 = _pick_columns(
        one_hot, right, left_columns[3], right_columns[3], limbs,
    )
    if what == "antiderivative":
        base = _pick_columns(one_hot, right, F_left, F_right, limbs)
        return base + s * (
            c0 + s * (c1 / 2.0 + s * (c2 / 3.0 + s * (c3 / 4.0)))
        )
    if order == 0:
        return c0 + s * (c1 + s * (c2 + s * c3))
    if order == 1:
        return c1 + s * (c2 * 2.0 + s * (c3 * 3.0))
    if order == 2:
        return c2 * 2.0 + s * (c3 * 6.0)
    if order == 3:
        return c3 * 6.0 + s * 0.0
    return s * 0.0


# --------------------------------------------------------------------------
# the interpolant


class Interpolant:
    """A piecewise cubic through ``(x, y)`` with exact derivatives and integrals.

    ``method`` is ``"linear"``, ``"natural_cubic"``, ``"pchip"`` or
    ``"akima"``; ``slopes`` supplies knot slopes directly (plain cubic
    Hermite).  Knots must be strictly increasing.  Points outside
    ``[x_0, x_{K-1}]`` continue the end pieces.

    ``limbs=1`` is ordinary float arithmetic.  ``limbs >= 2`` builds the
    coefficients exactly and evaluates in ``Precision[limbs]`` from the
    nearer knot, so every output is correctly rounded (see the module note).
    """

    def __init__(self, x: Any, y: Any, method: str = "natural_cubic",
                 slopes: Any = None, limbs: int = 1, build_limbs: int | None = None):
        self.x = _tensor(x).reshape(-1)
        self.y = _tensor(y).reshape(-1)
        self.method = method
        self.limbs = int(limbs)
        # Width of the rational build's components.  Wide enough that the
        # component products the build forms stay exact (see _build_exact).
        self.build_limbs = int(build_limbs) if build_limbs is not None else max(self.limbs + 2, 4)
        if self.limbs > 1:
            self._build_exact(slopes)
        h, d = _secants(self.x, self.y)
        if method == "linear" and slopes is None:
            zero = h * 0.0
            self.coefficients = (self.y[:-1], d, zero, zero)
        else:
            if slopes is None:
                if method not in _SLOPE_RULES:
                    raise ValueError(
                        f"unknown interpolation method {method!r}; one of "
                        f"{['linear', *_SLOPE_RULES]}")
                slopes = _SLOPE_RULES[method](self.x, self.y)
            self.slopes = _tensor(slopes).reshape(-1)
            m0, m1 = self.slopes[:-1], self.slopes[1:]
            self.coefficients = (
                self.y[:-1],
                m0,
                (3.0 * d - 2.0 * m0 - m1) / h,
                (m0 + m1 - 2.0 * d) / (h * h),
            )
        self.widths = h
        c0, c1, c2, c3 = self.coefficients
        pieces = c0 * h + c1 * h ** 2 / 2.0 + c2 * h ** 3 / 3.0 + c3 * h ** 4 / 4.0
        # F at each interval's left knot: 0, then the running sum of pieces.
        self.cumulative = AbstractTensor.concat([pieces[:1] * 0.0, pieces.cumsum(0)[:-1]], dim=0)
        self.total = pieces.sum()

    # -- the exact build (limbs >= 2) --------------------------------------

    def _build_exact(self, slopes: Any) -> None:
        """The coefficients, built on the rational system and read out at ``limbs``.

        The data enter as ``RationalPrecision`` (each float is exactly a
        quotient over 1); the slope rule, the secants, the Hermite
        coefficients and the running antiderivative are formed as deferred
        quotients -- the same generic slope rules the float path runs -- and
        each is divided exactly once, at readout (``limb_terms``), to the
        nearest ``limbs``-limb value.  ``build_limbs`` is the components'
        width: the products a build forms must fit it for the quotients to be
        exact (a component that would leave the element is refused by the
        rational limits, not rounded).

        Per interval ``i``: the left form ``(y_i, m_i, c2, c3)`` and the same
        piece about the RIGHT knot, ``(y_{i+1}, m_{i+1}, c2 + 3 c3 h, c3)`` --
        the data and slopes at both knots are taken as they are, never
        recomputed -- plus the running antiderivative at each knot.
        """

        from .extended_precision import RationalPrecision, concat

        width = self.build_limbs
        X = RationalPrecision.of(self.x, width)
        Y = RationalPrecision.of(self.y, width)
        h, d = _secants(X, Y)
        if self.method == "linear" and slopes is None:
            m0, m1 = d, d
            c2 = d * 0.0
            c3 = d * 0.0
        else:
            if slopes is not None:
                knot = RationalPrecision.of(_tensor(slopes).reshape(-1), width)
            elif self.method in _SLOPE_RULES:
                knot = _SLOPE_RULES[self.method](X, Y)
            else:
                raise ValueError(f"unknown interpolation method {self.method!r}")
            m0, m1 = knot[:-1], knot[1:]
            c2 = (d * 3.0 - m0 * 2.0 - m1) / h
            c3 = (m0 + m1 - d * 2.0) / (h * h)
        left = (Y[:-1], m0, c2, c3)
        right = (Y[1:], m1, c2 + c3 * h * 3.0, c3)
        pieces = h * (left[0] + h * (left[1] / 2.0 + h * (c2 / 3.0 + h * (c3 / 4.0))))
        running = concat([pieces[:1] * 0.0, pieces.cumsum(0)], dim=0)      # F at every knot
        n = self.limbs
        self._left = [column.limb_terms(n) for column in left]
        self._right = [column.limb_terms(n) for column in right]
        self._F_left = running[:-1].limb_terms(n)
        self._F_right = running[1:].limb_terms(n)

    def _locate_nearer(self, points: AbstractTensor):
        """``(one_hot, right, s)``: the interval, whether its right knot is the
        nearer one (0/1), and the exact offset from that knot as a Precision."""
        return _locate_nearer_columns(self.x, self.limbs, points)

    def _pick(self, one_hot, right, left_limbs, right_limbs):
        """One coefficient per point, limb by limb, from the nearer knot's form."""
        return _pick_columns(
            one_hot, right, left_limbs, right_limbs, self.limbs,
        )

    def _wide(self, points: AbstractTensor, what: str, order: int = 0):
        return _wide_from_columns(
            self.x,
            self.limbs,
            points,
            what,
            order,
            self._left,
            self._right,
            self._F_left,
            self._F_right,
        )

    # -- locating points ---------------------------------------------------

    def _locate(self, points: AbstractTensor):
        """``(one_hot (P, K-1), s (P,))`` for flattened points."""

        flat = points.reshape(-1)
        count = int(self.x.shape[0])
        inner = self.x[1:-1]                                   # interior knots
        index = (flat.reshape(-1, 1) >= inner.reshape(1, -1)).sum(1) * 1.0   # 0..K-2
        intervals = AbstractTensor.arange(count - 1) * 1.0
        one_hot = (index.reshape(-1, 1) == intervals.reshape(1, -1)) * 1.0
        left = (one_hot * self.x[:-1].reshape(1, -1)).sum(1)
        return one_hot, flat - left

    def _select(self, one_hot, per_interval):
        return (one_hot * per_interval.reshape(1, -1)).sum(1)

    # -- calculus ----------------------------------------------------------

    def __call__(self, points: Any) -> AbstractTensor:
        return self.evaluate(points)

    def evaluate(self, points: Any) -> AbstractTensor:
        points = _tensor(points)
        if self.limbs > 1:
            return self._wide(points, "value").collapse().reshape(*tuple(points.shape))
        one_hot, s = self._locate(points)
        c0, c1, c2, c3 = (self._select(one_hot, c) for c in self.coefficients)
        return (c0 + s * (c1 + s * (c2 + s * c3))).reshape(*tuple(points.shape))

    def derivative(self, points: Any, order: int = 1) -> AbstractTensor:
        """Exact derivative of the given order (0 to 3; zero beyond)."""

        points = _tensor(points)
        if self.limbs > 1:
            return self._wide(points, "derivative", order).collapse().reshape(*tuple(points.shape))
        one_hot, s = self._locate(points)
        c0, c1, c2, c3 = (self._select(one_hot, c) for c in self.coefficients)
        if order == 0:
            value = c0 + s * (c1 + s * (c2 + s * c3))
        elif order == 1:
            value = c1 + s * (2.0 * c2 + s * 3.0 * c3)
        elif order == 2:
            value = 2.0 * c2 + 6.0 * c3 * s
        elif order == 3:
            value = 6.0 * c3 + s * 0.0
        else:
            value = s * 0.0
        return value.reshape(*tuple(points.shape))

    def antiderivative(self, points: Any) -> AbstractTensor:
        """``F(x) = integral from x_0 to x`` of the interpolant, exactly."""

        points = _tensor(points)
        if self.limbs > 1:
            return self._wide(points, "antiderivative").collapse().reshape(*tuple(points.shape))
        one_hot, s = self._locate(points)
        c0, c1, c2, c3 = (self._select(one_hot, c) for c in self.coefficients)
        base = self._select(one_hot, self.cumulative)
        value = base + s * (c0 + s * (c1 / 2.0 + s * (c2 / 3.0 + s * c3 / 4.0)))
        return value.reshape(*tuple(points.shape))

    def integral(self, lower: Any, upper: Any) -> AbstractTensor:
        if self.limbs > 1:
            lower, upper = _tensor(lower), _tensor(upper)
            return (self._wide(upper, "antiderivative")
                    - self._wide(lower, "antiderivative")).collapse().reshape(*tuple(upper.shape))
        return self.antiderivative(upper) - self.antiderivative(lower)

    # -- built on the antiderivative -----------------------------------------

    def bin_averages(self, edges: Any) -> AbstractTensor:
        """Mean over each bin ``[edges[j], edges[j+1]]``: area-preserving resampling."""

        edges = _tensor(edges).reshape(-1)
        if self.limbs > 1:
            from .extended_precision import Precision

            wide = self._wide(edges, "antiderivative")
            terms = wide.terms()
            upper = Precision([term[1:] for term in terms], self.limbs)
            lower = Precision([term[:-1] for term in terms], self.limbs)
            widths = (Precision.of(edges[1:], self.limbs)
                      - Precision.of(edges[:-1], self.limbs))
            return ((upper - lower) / widths).collapse()
        F = self.antiderivative(edges)
        return (F[1:] - F[:-1]) / (edges[1:] - edges[:-1])

    def window_maxima(self, width: Any, starts: Any) -> AbstractTensor:
        """The largest mean over ``[t, t + width]`` among the given starts ``t``."""

        starts = _tensor(starts).reshape(-1)
        width = _tensor(width)
        if self.limbs > 1:
            from .extended_precision import Precision

            ends = starts + width
            spans = Precision.of(ends, self.limbs) - Precision.of(starts, self.limbs)
            means = (self._wide(ends, "antiderivative")
                     - self._wide(starts, "antiderivative")) / spans
            return means.collapse().max()
        means = (self.antiderivative(starts + width) - self.antiderivative(starts)) / width
        return means.max()

    def resampled(self, points: Any, method: str | None = None) -> "Interpolant":
        """A new interpolant through this one's values at ``points``
        (the dense-map pass: resample, then re-fit with ``method``)."""

        points = _tensor(points).reshape(-1)
        return Interpolant(points, self.evaluate(points), method or self.method,
                           limbs=self.limbs)


__all__ = (
    "Interpolant",
    "akima_slopes",
    "natural_cubic_slopes",
    "pchip_slopes",
)
