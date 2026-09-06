"""Is bit-exact reachable? Evaluate one core in double-double and count.

``demo_noise_floor_all`` compares every operator against a 40-digit reference
rounded to double -- which IS the correctly-rounded result. libm's ``sqrt``
scores -6004 dB there because its error is exactly zero, so that chart already
shows what bit-exact looks like: silence, not a smaller number.

The question this answers is whether our own cores can get there, and what
stands in the way. The suspicion is that the COEFFICIENTS are already good
enough and the loss is in evaluating them, because a Horner chain in double
commits a rounding error at every step and there are twenty of them.

So evaluate the same series three ways on the same points:

* ``double``  -- an ordinary Horner chain, what the surface does today;
* ``dd``      -- the identical series in double-double (Dekker/Knuth
  error-free transformations), rounded once at the end;
* ``libm``    -- for scale.

If ``dd`` lands bit-exact on nearly every point, the coefficients were never
the problem and correct rounding is an evaluation-precision question, which is
exactly what Ziv's onion-peeling strategy is built on: compute with a known
error bound, test whether the bound straddles a rounding boundary, and escalate
only on the few points that fail.

Run::

    python -m tools.demo_bit_exact --size 4096
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Dekker's splitting constant: 2**27 + 1 for binary64.
SPLIT = 134217729.0


def two_sum(a, b):
    """Exact sum: returns (s, e) with a + b == s + e in exact arithmetic."""

    s = a + b
    shifted = s - a
    return s, (a - (s - shifted)) + (b - shifted)


def split(a):
    """Split a double into two halves with non-overlapping significands."""

    c = SPLIT * a
    high = c - (c - a)
    return high, a - high


def two_product(a, b):
    """Exact product: returns (p, e) with a * b == p + e exactly.

    numpy exposes no fused multiply-add, so this is Dekker's original
    splitting form rather than the one-line FMA version.
    """

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def dd_add(ah, al, bh, bl):
    """Add two double-doubles."""

    s, e = two_sum(ah, bh)
    e = e + (al + bl)
    return two_sum(s, e)


def dd_mul(ah, al, bh, bl):
    """Multiply two double-doubles."""

    p, e = two_product(ah, bh)
    e = e + (ah * bl + al * bh)
    return two_sum(p, e)


def dd_horner(xh, xl, coefficients):
    """Horner in double-double over double-double coefficients."""

    rh = np.zeros_like(xh)
    rl = np.zeros_like(xh)
    for ch, cl in reversed(coefficients):
        rh, rl = dd_mul(rh, rl, xh, xl)
        rh, rl = dd_add(rh, rl, ch, cl)
    return rh, rl


def sine_coefficients(terms):
    """P in sin(y) = y * P(y*y), coefficients as double-double pairs.

    (-1)**k / (2k+1)! is exact in arbitrary precision, so each coefficient is
    split into a leading double and the remainder it could not hold.
    """

    import mpmath

    pairs = []
    with mpmath.workdps(60):
        for k in range(terms):
            value = mpmath.mpf(-1) ** k / mpmath.factorial(2 * k + 1)
            high = float(value)
            pairs.append((high, float(value - mpmath.mpf(high))))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--tone", type=int, default=17)
    parser.add_argument("--terms", type=int, default=15)
    arguments = parser.parse_args(argv)

    import mpmath

    # Stay inside sin's own core so no argument reduction is involved: this
    # measures the EVALUATION, and a reduction would smuggle in its own error.
    limit = 0.7
    sample = np.arange(arguments.size)
    x = limit * np.cos(2.0 * math.pi * arguments.tone * sample / arguments.size)

    with mpmath.workdps(40):
        exact = np.array([float(mpmath.sin(mpmath.mpf(float(v)))) for v in x])

    coefficients = sine_coefficients(arguments.terms)
    square = x * x

    # The surface's form, in plain double.
    plain = np.zeros_like(x)
    for high, _ in reversed(coefficients):
        plain = plain * square + high
    plain = x * plain

    # The identical series, evaluated in double-double and rounded once.
    sh, sl = two_product(x, x)
    ph, pl = dd_horner(sh, sl, coefficients)
    rh, rl = dd_mul(ph, pl, x, np.zeros_like(x))
    double_double = rh + rl

    theirs = np.sin(x)

    print(f"sin on {arguments.size} points in [-{limit}, {limit}], "
          f"{arguments.terms} terms, reference at 40 digits")
    print(f"{'':14s} {'bit-exact':>12s} {'max ulp':>10s} {'noise dB':>10s}")
    scale = float(np.sqrt(np.mean(exact * exact)))
    for label, produced in (("double", plain), ("double-double", double_double),
                            ("libm", theirs)):
        difference = produced - exact
        # ulp measured against the reference's own exponent, not a global one.
        ulp = np.abs(difference) / np.spacing(np.abs(exact))
        matched = float(np.mean(produced == exact)) * 100.0
        rms = float(np.sqrt(np.mean(difference * difference)))
        noise = 20.0 * math.log10(max(rms, 1e-300) / scale)
        print(f"{label:14s} {matched:11.4f}% {ulp.max():10.3f} {noise:10.1f}")

    disagree = int(np.sum(double_double != exact))
    print(f"\ndouble-double disagrees with correctly-rounded on "
          f"{disagree} of {arguments.size} points")
    if disagree:
        index = np.argmax(np.abs(double_double - exact))
        print(f"  worst at x = {x[index]:.17g}")
        print(f"    ours {double_double[index]!r}")
        print(f"    true {exact[index]!r}")
        print("  These are the hardest-to-round points -- the ones Ziv's test"
              " catches and escalates. Their count is the whole cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
