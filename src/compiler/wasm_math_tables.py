"""Every function WebAssembly does not have, as data it can carry.

WebAssembly's float instructions stop at abs/neg/ceil/floor/trunc/nearest/
sqrt/min/max. Everything else a numeric program reaches for -- the
trigonometric family, the hyperbolics, exponentials, logarithms -- has no
opcode at all. A backend has three honest options: refuse, evaluate a
polynomial, or carry a table. This module builds the tables, and can build
the polynomials instead when a caller prefers them.

Both are approximations, which is exactly why they are declared. Each entry
reports the absolute error bound it was sized to and the domain it holds
over, and a caller can read those before deciding. The refusal elsewhere in
this backend is aimed at silently substituting a guess for what was asked
for; a bounded, measured, stated approximation is a different thing.

Sizing follows llvm_signal_math, which does this for sine on the LLVM path:
linear interpolation error over a step h is bounded by max|f''| * h^2 / 8.
Rather than deriving that maximum by hand for a dozen functions and getting
one of them wrong, it is measured numerically on a dense probe of the
domain, then carried with a safety factor.

Domains are chosen where a function is worth tabulating and clamped or
wrapped outside it:

* periodic functions wrap, so they stay exact for any argument, which
  matters when the argument is a frame counter that grows without bound;
* saturating functions (tanh, atan) clamp, because past the domain they are
  constant to more precision than the table has anyway;
* the rest clamp and say so in ``domain``, because a caller feeding them
  outside it wants to know rather than to be quietly given the endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

DEFAULT_EPSILON = 1.0e-6
TAU = 2.0 * math.pi

# Sampling used to measure curvature. Dense enough that a peak between
# samples is caught, and the result is inflated below regardless.
_PROBE = 4096
# max|f''| is measured, not derived, so it is scaled up before being trusted.
_CURVATURE_SAFETY = 1.35


@dataclass(frozen=True)
class MathFunction:
    """One function, and how to approximate it."""

    name: str
    evaluate: Callable[[float], float]
    lower: float
    upper: float
    periodic: bool = False
    # Set when the domain is a restriction rather than the whole story, so a
    # caller can see the limit instead of discovering it in the output.
    note: str = ""
    # Taylor/Maclaurin terms, for the polynomial alternative. Absent when a
    # series is not a sensible way to reach the function over this domain.
    series: Callable[[int], Sequence[float]] | None = None
    series_domain: tuple[float, float] | None = None


def _maclaurin_sin(terms: int) -> list[float]:
    # sin x = sum (-1)^k x^(2k+1)/(2k+1)!
    return [
        ((-1.0) ** k) / math.factorial(2 * k + 1) for k in range(terms)
    ]


def _maclaurin_cos(terms: int) -> list[float]:
    return [((-1.0) ** k) / math.factorial(2 * k) for k in range(terms)]


def _maclaurin_exp(terms: int) -> list[float]:
    return [1.0 / math.factorial(k) for k in range(terms)]


def _maclaurin_atan(terms: int) -> list[float]:
    # Converges only for |x| <= 1; the domain below reflects that.
    return [((-1.0) ** k) / (2 * k + 1) for k in range(terms)]


# The catalogue. Anything absent has no table and stays refused rather than
# being approximated by something that happens to be nearby.
FUNCTIONS: dict[str, MathFunction] = {
    "sin": MathFunction("sin", math.sin, 0.0, TAU, periodic=True,
                        series=_maclaurin_sin, series_domain=(-math.pi, math.pi)),
    "cos": MathFunction("cos", math.cos, 0.0, TAU, periodic=True,
                        series=_maclaurin_cos, series_domain=(-math.pi, math.pi)),
    # tan is not tabulated: it has poles inside any interval worth covering,
    # and no bounded table can describe a function that is unbounded there.
    # A program wanting it should divide sin by cos and decide for itself
    # what to do near the pole.
    "tanh": MathFunction("tanh", math.tanh, -8.0, 8.0,
                         note="saturates; |x| > 8 is 1 to within 3e-7"),
    # Saturating like tanh, but approaching its limits half as fast, so the
    # domain is twice as wide: exp(-16) is 1.1e-7, under the table's own
    # epsilon, whereas clamping at 8 would already be 3.4e-4 short.
    "sigmoid": MathFunction("sigmoid", lambda x: 0.5 * (1.0 + math.tanh(0.5 * x)),
                            -16.0, 16.0,
                            note="saturates; |x| > 16 is 0 or 1 to within 1.2e-7"),
    "sinh": MathFunction("sinh", math.sinh, -6.0, 6.0,
                         note="clamped at |x| = 6; grows without bound beyond"),
    "cosh": MathFunction("cosh", math.cosh, -6.0, 6.0,
                         note="clamped at |x| = 6; grows without bound beyond"),
    "asinh": MathFunction("asinh", math.asinh, -8.0, 8.0,
                          note="clamped at |x| = 8"),
    "atan": MathFunction("atan", math.atan, -8.0, 8.0,
                         note="saturates towards +/-pi/2; clamped at |x| = 8",
                         series=_maclaurin_atan, series_domain=(-0.7, 0.7)),
    # asin/acos have infinite curvature at the endpoints, so the table is cut
    # just short of them. A caller needing the last sliver needs a different
    # method, not a finer table.
    "asin": MathFunction("asin", math.asin, -0.999, 0.999,
                         note="endpoints excluded; curvature is unbounded at +/-1"),
    "acos": MathFunction("acos", math.acos, -0.999, 0.999,
                         note="endpoints excluded; curvature is unbounded at +/-1"),
    "acosh": MathFunction("acosh", math.acosh, 1.0001, 12.0,
                          note="defined for x >= 1; curvature is unbounded at 1"),
    "atanh": MathFunction("atanh", math.atanh, -0.999, 0.999,
                          note="endpoints excluded; diverges at +/-1"),
    "exp2": MathFunction("exp2", lambda x: 2.0 ** x, -40.0, 8.0,
                         note="clamped; 2^-40 is already below single-step "
                              "resolution for most uses"),
    "exp": MathFunction("exp", math.exp, -30.0, 6.0,
                        note="clamped at 6; use exp2 for a wider range",
                        series=_maclaurin_exp, series_domain=(-2.0, 2.0)),
    # The logarithms start at 1/4 rather than at zero. Their curvature is
    # 1/x^2, so a domain reaching towards zero forces a table that is
    # enormous and still worst exactly where it is needed: at 1e-6 the
    # table wanted 64 MB. A caller with a smaller argument should scale it
    # into range and add the constant back -- log(x*2^k) = log(x) + k*ln2 --
    # which is exact and costs one multiply, rather than being handed a
    # table that quietly cannot deliver its stated accuracy there.
    "log": MathFunction("log", math.log, 0.25, 64.0,
                        note="x <= 0 undefined; below 1/4, scale by a power "
                             "of two and add k*ln2"),
    "log2": MathFunction("log2", math.log2, 0.25, 64.0,
                         note="x <= 0 undefined; below 1/4, scale by a power "
                              "of two and add k"),
}


def _power_of_two_ceiling(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def measure_curvature(function: MathFunction) -> float:
    """Largest |f''| seen across the domain, by second difference.

    Measured rather than derived: a dozen hand-derived bounds is a dozen
    chances to be wrong about one, and being wrong here means a table that
    quietly misses its stated accuracy.
    """

    lower, upper = function.lower, function.upper
    step = (upper - lower) / _PROBE
    worst = 0.0
    for index in range(1, _PROBE):
        x = lower + step * index
        try:
            second = (
                function.evaluate(x - step)
                - 2.0 * function.evaluate(x)
                + function.evaluate(x + step)
            ) / (step * step)
        except (ValueError, OverflowError):
            continue
        if math.isfinite(second):
            worst = max(worst, abs(second))
    return max(worst, 1.0e-9)


@dataclass(frozen=True)
class MathTable:
    """A sampled function, with the accuracy it was built to."""

    name: str
    values: tuple[float, ...]
    lower: float
    upper: float
    periodic: bool
    bound: float
    curvature: float
    note: str = ""

    @property
    def intervals(self) -> int:
        return len(self.values) - 1

    @property
    def byte_length(self) -> int:
        return len(self.values) * 8

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "intervals": self.intervals,
            "lower": self.lower,
            "upper": self.upper,
            "periodic": self.periodic,
            "bound": self.bound,
            "curvature": self.curvature,
            "bytes": self.byte_length,
            "note": self.note,
        }


def build_table(name: str, epsilon: float = DEFAULT_EPSILON) -> MathTable:
    """Sample ``name`` finely enough that interpolation stays under
    ``epsilon``."""

    function = FUNCTIONS.get(name)
    if function is None:
        raise KeyError(
            f"no table is defined for {name!r}; the catalogue is "
            f"{sorted(FUNCTIONS)}"
        )
    curvature = measure_curvature(function) * _CURVATURE_SAFETY
    span = function.upper - function.lower
    maximum_step = math.sqrt(8.0 * epsilon / curvature)
    required = max(4, math.ceil(span / maximum_step))
    intervals = _power_of_two_ceiling(required)
    step = span / intervals
    values = []
    for index in range(intervals + 1):
        x = function.lower + step * index
        try:
            values.append(float(function.evaluate(x)))
        except (ValueError, OverflowError):
            # Only reachable at a domain edge; hold the neighbour rather than
            # writing a NaN into a table someone will interpolate through.
            values.append(values[-1] if values else 0.0)
    return MathTable(
        name=name,
        values=tuple(values),
        lower=function.lower,
        upper=function.upper,
        periodic=function.periodic,
        bound=curvature * step * step / 8.0,
        curvature=curvature,
        note=function.note,
    )


def measure_error(table: MathTable, samples: int = 20001) -> float:
    """The error the table actually delivers, by comparing against the
    function it came from. The bound is a prediction; this is the result."""

    function = FUNCTIONS[table.name]
    span = table.upper - table.lower
    step = span / table.intervals
    worst = 0.0
    for index in range(samples):
        x = table.lower + span * index / (samples - 1)
        if table.periodic:
            reduced = x - math.floor((x - table.lower) / span) * span
        else:
            reduced = min(max(x, table.lower), table.upper)
        position = (reduced - table.lower) / step
        slot = min(int(position), table.intervals - 1)
        fraction = position - slot
        approximated = (
            table.values[slot]
            + (table.values[slot + 1] - table.values[slot]) * fraction
        )
        try:
            exact = function.evaluate(x)
        except (ValueError, OverflowError):
            continue
        if math.isfinite(exact):
            worst = max(worst, abs(approximated - exact))
    return worst


# --- the polynomial alternative -------------------------------------------


@dataclass(frozen=True)
class MathSeries:
    """A truncated Maclaurin series, for a caller who would rather evaluate
    than look up.

    A series costs no memory and reaches full precision near the origin, but
    it needs many terms far from it and diverges outside its radius. A table
    is flat in cost across its whole domain. Which is better is a property of
    the program, not of this module, so both are offered.
    """

    name: str
    coefficients: tuple[float, ...]
    # Powers the coefficients multiply: sin is odd, cos even, exp all.
    powers: tuple[int, ...]
    lower: float
    upper: float
    bound: float

    def evaluate(self, x: float) -> float:
        return sum(c * x ** p for c, p in zip(self.coefficients, self.powers))

    def horner(self) -> tuple[int, tuple[float, ...]]:
        """The series as (leading power, coefficients in x^stride).

        sin and atan are odd and cos is even, so each is a polynomial in
        x^2 multiplied by x^(leading power). Evaluating that way is the
        form llvm_signal_math emits, and it costs one multiply per term
        instead of a power per term.
        """

        stride = 2 if self.name in ("sin", "cos", "atan") else 1
        leading = self.powers[0]
        for index, power in enumerate(self.powers):
            if power != leading + stride * index:
                raise ValueError(
                    f"{self.name} powers are not a regular progression; "
                    "Horner form does not apply"
                )
        return leading, self.coefficients


def build_series(name: str, epsilon: float = DEFAULT_EPSILON) -> MathSeries:
    """Take terms until the remainder is provably under ``epsilon``.

    Same reasoning llvm_signal_math._continuous_terms uses on the LLVM path,
    so the two agree on what a given epsilon buys. The series here are
    alternating with decreasing terms over their stated domain, so the first
    omitted term bounds the remainder -- a proof rather than a measurement,
    which is what lets the answer be trusted between the sample points a
    measurement would have checked.

    exp is the exception: its terms are all positive, so the alternating
    bound does not apply and the Lagrange remainder is used instead.
    """

    function = FUNCTIONS.get(name)
    if function is None or function.series is None:
        raise KeyError(
            f"no series is defined for {name!r}; "
            f"{sorted(n for n, f in FUNCTIONS.items() if f.series)} have one"
        )
    lower, upper = function.series_domain or (function.lower, function.upper)
    radius = max(abs(lower), abs(upper))
    powers_for = {
        "sin": lambda k: 2 * k + 1,
        "cos": lambda k: 2 * k,
        "atan": lambda k: 2 * k + 1,
        "exp": lambda k: k,
    }[name]

    def remainder(terms: int) -> float:
        if name == "exp":
            # |R_n| <= e^r * r^n / n! on [-r, r].
            return math.exp(radius) * radius ** terms / math.factorial(terms)
        power = powers_for(terms)
        if name == "atan":
            # The omitted term is r^p / p, not r^p / p!.
            return radius ** power / power
        return radius ** power / math.factorial(power)

    for terms in range(1, 41):
        bound = remainder(terms)
        if bound <= epsilon:
            coefficients = tuple(function.series(terms))
            return MathSeries(
                name=name,
                coefficients=coefficients,
                powers=tuple(powers_for(k) for k in range(terms)),
                lower=lower,
                upper=upper,
                bound=bound,
            )
    raise ValueError(
        f"{name} did not reach {epsilon} within 40 terms over "
        f"[{lower}, {upper}]; a table is the better tool here"
    )


TABULATED = frozenset(FUNCTIONS)
SERIES_CAPABLE = frozenset(n for n, f in FUNCTIONS.items() if f.series)


__all__ = [
    "DEFAULT_EPSILON",
    "FUNCTIONS",
    "MathFunction",
    "MathSeries",
    "MathTable",
    "SERIES_CAPABLE",
    "TABULATED",
    "build_series",
    "build_table",
    "measure_curvature",
    "measure_error",
]
