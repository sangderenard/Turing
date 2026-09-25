"""Error-well sweep for ``src.common.tensors.interpolants`` against an exact oracle.

Test infrastructure only.  The oracle is host ``fractions.Fraction``
arithmetic: the same slope rules and coefficient formulas evaluated exactly,
so "correct" means the result equals the correctly rounded double of the
exact interpolant of the float data (``float(Fraction)`` rounds correctly).

``--inject-oracle-build`` fills ``Interpolant._build_exact`` with the
Fraction build so the module's own ``limbs >= 2`` evaluation path
(Precision, nearer-knot anchoring, limb-wise selection) can be measured
before the rational build exists.  The injected build does not compile and
is never shipped; it is the yardstick the rational build must match.

Usage::

    python tools/interpolant_error_wells.py                       # float path
    python tools/interpolant_error_wells.py --inject-oracle-build # limbs=2 path
    python tools/interpolant_error_wells.py --inject-oracle-build --limbs 3 --trials 24

Measured 2026-09-24 (72 trials): float path 3-600 ULP worst per method;
limbs=2 with the oracle build, 0 misses in 69,120 outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common.tensors import AbstractTensor  # noqa: E402
import src.common.tensors.numpy_backend  # noqa: E402,F401
from src.common.tensors.interpolants import Interpolant  # noqa: E402


# ---------------------------------------------------------------- exact oracle

def _sign(value) -> int:
    return (value > 0) - (value < 0)


def exact_secants(x, y):
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    return h, [(y[i + 1] - y[i]) / h[i] for i in range(len(h))]


def exact_natural_cubic(x, y):
    h, d = exact_secants(x, y)
    count = len(x)
    below = [Fraction(0)] * count
    diag = [Fraction(1)] * count
    above = [Fraction(0)] * count
    rhs = [Fraction(0)] * count
    for i in range(1, count - 1):
        below[i], diag[i], above[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6 * (d[i] - d[i - 1])
    for i in range(1, count):
        factor = below[i] / diag[i - 1]
        diag[i] -= factor * above[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    second = [Fraction(0)] * count
    second[-1] = rhs[-1] / diag[-1]
    for i in range(count - 2, -1, -1):
        second[i] = (rhs[i] - above[i] * second[i + 1]) / diag[i]
    slopes = [d[i] - h[i] * (2 * second[i] + second[i + 1]) / 6 for i in range(count - 1)]
    return slopes + [d[-1] + h[-1] * (second[-2] + 2 * second[-1]) / 6]


def exact_pchip(x, y):
    h, d = exact_secants(x, y)
    slopes = [Fraction(0)] * len(x)
    for i in range(1, len(x) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def edge(ha, hb, da, db):
        estimate = ((2 * ha + hb) * da - ha * db) / (ha + hb)
        if _sign(estimate) != _sign(da):
            return Fraction(0)
        if _sign(da) != _sign(db) and abs(estimate) > 3 * abs(da):
            return 3 * da
        return estimate

    slopes[0] = edge(h[0], h[1], d[0], d[1])
    slopes[-1] = edge(h[-1], h[-2], d[-1], d[-2])
    return slopes


def exact_akima(x, y):
    _h, d = exact_secants(x, y)
    before1 = 2 * d[0] - d[1]
    after1 = 2 * d[-1] - d[-2]
    ext = [2 * before1 - d[0], before1, *d, after1, 2 * after1 - d[-1]]
    slopes = []
    for i in range(len(x)):
        dm2, dm1, d0, dp1 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        w_left, w_right = abs(dp1 - d0), abs(dm1 - dm2)
        total = w_left + w_right
        slopes.append((dm1 + d0) / 2 if total == 0 else (w_left * dm1 + w_right * d0) / total)
    return slopes


EXACT_SLOPES = {"natural_cubic": exact_natural_cubic, "pchip": exact_pchip, "akima": exact_akima}


class ExactInterpolant:
    """The exact interpolant of the float data: coefficients, values, calculus."""

    def __init__(self, x, y, method):
        self.x = [Fraction(v) for v in x]
        self.y = [Fraction(v) for v in y]
        h, d = exact_secants(self.x, self.y)
        self.h = h
        if method == "linear":
            self.left = [(self.y[i], d[i], Fraction(0), Fraction(0)) for i in range(len(h))]
        else:
            m = EXACT_SLOPES[method](self.x, self.y)
            self.left = [(self.y[i], m[i], (3 * d[i] - 2 * m[i] - m[i + 1]) / h[i],
                          (m[i] + m[i + 1] - 2 * d[i]) / h[i] ** 2) for i in range(len(h))]
        self.right = [(c0 + c1 * hh + c2 * hh ** 2 + c3 * hh ** 3, c1 + 2 * c2 * hh + 3 * c3 * hh ** 2,
                       c2 + 3 * c3 * hh, c3) for (c0, c1, c2, c3), hh in zip(self.left, h)]
        self.F = [Fraction(0)]
        for (c0, c1, c2, c3), hh in zip(self.left, h):
            self.F.append(self.F[-1] + c0 * hh + c1 * hh ** 2 / 2 + c2 * hh ** 3 / 3 + c3 * hh ** 4 / 4)

    def _piece(self, p):
        i = sum(1 for k in self.x[1:-1] if p >= k)
        return i, p - self.x[i]

    def value(self, p):
        i, s = self._piece(Fraction(p)); c0, c1, c2, c3 = self.left[i]
        return c0 + c1 * s + c2 * s ** 2 + c3 * s ** 3

    def d1(self, p):
        i, s = self._piece(Fraction(p)); _c0, c1, c2, c3 = self.left[i]
        return c1 + 2 * c2 * s + 3 * c3 * s ** 2

    def antiderivative(self, p):
        i, s = self._piece(Fraction(p)); c0, c1, c2, c3 = self.left[i]
        return self.F[i] + c0 * s + c1 * s ** 2 / 2 + c2 * s ** 3 / 3 + c3 * s ** 4 / 4


def _limb_columns(values, limbs):
    from src.common.tensors.signal_symbolic import limb_decomposition

    rows = [limb_decomposition(value, limbs) for value in values]
    return [AbstractTensor.get_tensor([row[j] for row in rows]) * 1.0 for j in range(limbs)]


def inject_oracle_build():
    """Test-only: give ``Interpolant(limbs >= 2)`` the exact Fraction build."""

    def build(self, slopes):
        if slopes is not None:
            raise NotImplementedError("oracle build takes method slopes only")
        exact = ExactInterpolant(self.x.tolist(), self.y.tolist(), self.method)
        n = self.limbs
        self._left = [_limb_columns(column, n) for column in zip(*exact.left)]
        self._right = [_limb_columns(column, n) for column in zip(*exact.right)]
        self._F_left = _limb_columns(exact.F[:-1], n)
        self._F_right = _limb_columns(exact.F[1:], n)

    Interpolant._build_exact = build


# ---------------------------------------------------------------- sweep

def _ulp(got, exact):
    ef = float(exact)
    if ef == 0.0:
        return float("inf") if got != 0.0 else 0.0
    return float(abs(Fraction(float(got)) - exact) / Fraction(np.spacing(abs(ef))))


def _dataset(rng, k, spacing, shape):
    if spacing == "uniform":
        x = np.linspace(0.0, 10.0, k)
    elif spacing == "random":
        x = np.cumsum(rng.uniform(0.05, 2.0, k))
    else:
        x = np.cumsum(np.where(rng.uniform(size=k) < 0.5, rng.uniform(1e-3, 1e-2, k), rng.uniform(0.5, 2.0, k)))
    if shape == "smooth":
        y = np.sin(x) + 0.3 * np.cos(2.7 * x)
    elif shape == "noisy":
        y = np.sin(x) + rng.normal(0, 0.5, k)
    elif shape == "steps":
        y = np.floor(rng.uniform(0, 4, k))
    elif shape == "offset":
        x = x + 1e6
        y = 7.0 + 1e-8 * np.sin(x - 1e6)
    elif shape == "dynamic":
        y = np.sin(x) * 10.0 ** rng.uniform(-6, 6, k)
    else:
        y = (x - x.mean()) * (1 + 0.1 * np.sin(3 * x))
    return x, y


def sweep(trials: int, limbs: int, seed: int = 20260924) -> int:
    rng = np.random.default_rng(seed)
    T = AbstractTensor.tensor
    arr = lambda t: np.asarray(t.tolist(), dtype=np.float64)  # noqa: E731
    misses, worst, checked = defaultdict(list), defaultdict(float), Counter()
    spacings = ("uniform", "random", "clustered")
    shapes = ("smooth", "noisy", "steps", "offset", "dynamic", "zero_crossing")
    started = time.perf_counter()
    for trial in range(trials):
        k = int(rng.integers(4, 24))
        spacing, shape = spacings[trial % 3], shapes[(trial // 3) % 6]
        x, y = _dataset(rng, k, spacing, shape)
        inside = rng.uniform(x[0], x[-1], 40)
        near = np.concatenate([x[1:-1] + np.spacing(x[1:-1]), x[1:-1] - np.spacing(x[1:-1])])
        points = np.sort(np.concatenate([inside, x, near]))
        edges = np.sort(np.concatenate([[x[0], x[-1]], rng.uniform(x[0], x[-1], 7)]))
        for method in ("linear", "natural_cubic", "pchip", "akima"):
            exact = ExactInterpolant(x, y, method)
            ip = Interpolant(T(x), T(y), method, limbs=limbs)
            got = {"value": arr(ip.evaluate(T(points))), "d1": arr(ip.derivative(T(points), 1)),
                   "F": arr(ip.antiderivative(T(points))), "bins": arr(ip.bin_averages(T(edges)))}
            want = {"value": [exact.value(p) for p in points], "d1": [exact.d1(p) for p in points],
                    "F": [exact.antiderivative(p) for p in points],
                    "bins": [(exact.antiderivative(b) - exact.antiderivative(a)) / (Fraction(b) - Fraction(a))
                             for a, b in zip(edges[:-1], edges[1:])]}
            where = {"value": points, "d1": points, "F": points, "bins": edges[:-1]}
            for out in got:
                for g, e, p in zip(got[out], want[out], where[out]):
                    checked[(method, out)] += 1
                    u = _ulp(g, e)
                    worst[(method, out)] = max(worst[(method, out)], u)
                    if g != float(e):
                        misses[(method, out)].append((u, shape, spacing, k, float(p), float(e)))
    print(f"limbs={limbs}: swept {trials} datasets in {time.perf_counter() - started:.0f}s "
          "(a miss = result != correctly rounded double of the exact interpolant)")
    print(f"{'method':14s} {'output':6s} {'checked':>8s} {'misses':>7s} {'worst ULP':>10s}")
    for key in sorted(checked):
        print(f"{key[0]:14s} {key[1]:6s} {checked[key]:8d} {len(misses[key]):7d} {worst[key]:10.4g}")
    for key in sorted(misses):
        for u, shape, spacing, k, p, e in sorted(misses[key], reverse=True)[:5]:
            print(f"  miss {key[0]:14s} {key[1]:6s} {u:9.4g}  {shape:13s} {spacing:9s} K={k:2d} x={p:.17g} exact={e:.6g}")
    return sum(len(v) for v in misses.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trials", type=int, default=72)
    parser.add_argument("--limbs", type=int, default=1)
    parser.add_argument("--inject-oracle-build", action="store_true")
    args = parser.parse_args()
    limbs = args.limbs
    if args.inject_oracle_build:
        inject_oracle_build()
        limbs = max(limbs, 2)
    return 0 if sweep(args.trials, limbs) == 0 or limbs == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
