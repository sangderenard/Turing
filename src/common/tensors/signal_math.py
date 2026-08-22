"""AbstractTensor signal math: the whole trigonometric surface, baked.

This is the AbstractTensor correlate of
``accelerator_backends/llvm_signal_math.py``, and it is intended to become the
authority that module emits FROM rather than a second opinion beside it.

Why it exists
-------------
``AbstractTensor.sin`` dispatches ``_apply_operator("sin")``, which lowers to
the repository kernel ``unary_double`` with opcode 29, whose body is
``call double @sin(double)`` -- an extern resolved by the platform's libm.
Every packed trigonometry artifact in this tree, forward and reverse, borrowed
its digits that way. ``llvm_signal_math`` already owned real alternatives, but
as hand-written LLVM text reachable only from a tape path that is dead to the
compiler and from a torture-matrix oracle, so no deployment product ever used
it.

Authoring the cores in AbstractTensor fixes that at the root. An AbstractTensor
implementation captures into repository SSA like any other authored source, so
ONE definition reaches native, shader and wasm products alike and the baked
constants travel inside the captured graph. Backend agnosticism is a
consequence of where the source lives, not a feature bolted on afterwards.

The three axes
--------------
``epsilon``
    An absolute error target. Every family GROWS until its measured residual
    meets it, never to a set count. Growth stops at the first admitted size
    rather than the largest tried, because accuracy is not monotone in size --
    conditioning made a degree-10 structured sine (1.89e-15) worse than
    degree 8 (1.55e-15).

``family`` -- ``structured`` / ``polyspline`` / ``series`` / ``lut``
    ``structured`` carries a function's parity in the FORM of the expression,
    ``sin(y) = y*P(y*y)`` and ``cos(y) = Q(y*y)``. That is not a micro
    optimization; it is what makes the structural guarantees exact. Measured
    against a plain 8-segment polyspline sine::

        metric                  plain polyspline   structured
        parity                     3.331e-16       0.000e+00
        sin(0)                    -1.665e-16       +0 exact
        max|sin| - 1              +1.110e-15       -1.110e-16
        derivative jump at knots   6.771e-13       none (no knots)
        coefficients                      80              9

    An overshoot past 1 is not cosmetic: it hands ``asin``/``acos`` an
    out-of-domain argument and turns ``sqrt(1 - s*s)`` into a NaN.

    ``polyspline`` is the fallback where no parity exists. ``series`` is the
    definitional form and the value AUTHORITY. ``lut`` is a PRERUN of the
    series: its nodes are computed by the admitted series core, so there is
    one authority for what a function is and the table is only a decision
    about how to pay for it.

``mode`` -- ``direct`` / ``implied``
    ``implied`` forms the surface from the smallest primitive set by identity.
    It is the SMALLEST representation and a fair default when footprint rules,
    at the cost of error compounding through each identity. ``direct`` gives a
    method its own core wherever one can be fitted, so its error is bounded on
    its own terms, and it is the default -- an identity route loses parity
    exactly the way a plain polynomial does.

Reduce in cycles, not radians
-----------------------------
Radian reduction divides by a float TAU and loses a digit of the fraction per
decade of magnitude. Measured, ``sin(2*pi*c)``::

    cycles     radian-reduced     turn-reduced
     1e+03        6.140e-13        8.882e-16
     1e+06        2.218e-10        8.882e-16
     1e+09        2.178e-07        8.882e-16
     1e+12        5.256e-04        8.882e-16

The turn entry is flat at machine epsilon because ``t - floor(t)`` is exact at
every magnitude. spectral-analyzer's COMPLEX_OPTICAL_OPERATOR_CONTRACT states
the same rule for carrier phase. Signal work already holds its argument in
cycles (``f*t``); prefer the ``_turns`` entries and never enter radians.

Nothing here silently extrapolates: a core states the interval it was fitted
on and the reduction that maps an argument into it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .abstraction import AbstractTensor


FAMILIES = ("exact", "structured", "polyspline", "series", "lut")
MODES = ("direct", "implied")

#: Measured: exact coefficients reach 1 ulp where a fit floors at 7,
#: for one extra coefficient. A fit cannot go below its own residual
#: however much degree it is given.
DEFAULT_FAMILY = "exact"
DEFAULT_MODE = "direct"

TAU = 2.0 * math.pi
HALF_PI = 0.5 * math.pi
QUARTER_PI = 0.25 * math.pi
LN2 = math.log(2.0)
LN10 = math.log(10.0)

#: The smallest set that implies the whole surface, used by ``implied`` mode.
PRIMITIVES = ("sin", "cos", "atan", "exp", "log", "sqrt")


# --------------------------------------------------------------------------
# What each core is, and on what interval
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoreRange:
    low: float
    high: float
    #: Parity the fit may carry structurally, or None.
    structure: str | None
    #: Expansion point for a ``series`` core.
    centre: float
    note: str


CORE_RANGES: Mapping[str, CoreRange] = {
    "sin": CoreRange(-QUARTER_PI, QUARTER_PI, "odd", 0.0,
                     "octant reduction; cos serves the other half"),
    "cos": CoreRange(-QUARTER_PI, QUARTER_PI, "even", 0.0,
                     "octant reduction; |cos| >= sqrt(2)/2 here, no zero"),
    # Narrowed on the error map's evidence. The residual climbed toward the
    # interval's top (8.7 -> 24.0 ulp across eighths), and the exact series
    # simply does not converge on [0,1] -- 8.2e12 ulp at order 21. At
    # tan(pi/8) it reaches 0.84 ulp in EIGHTEEN coefficients, so the narrower
    # interval is cheaper as well as better. The classic two-way split, found
    # by measuring rather than by looking it up.
    "atan": CoreRange(0.0, math.tan(math.pi / 8.0), "odd", 0.0,
                      "x -> (x-1)/(x+1) above tan(pi/8), 1/x above one"),
    "asin": CoreRange(-0.5, 0.5, "odd", 0.0, "half-angle reduction above 1/2"),
    "exp": CoreRange(-0.5 * LN2, 0.5 * LN2, None, 0.0, "x = k*ln2 + r"),
    "expm1": CoreRange(-0.5 * LN2, 0.5 * LN2, "factored", 0.0,
                       "own core; root at 0 factored out"),
    "log": CoreRange(math.sqrt(0.5), math.sqrt(2.0), None, 1.0,
                     "mantissa band around 1"),
    "log1p": CoreRange(-0.25, 0.25, "factored", 0.0,
                       "own core; root at 0 factored out. The top matters: "
                       "this series has radius 1, so its remainder decays "
                       "like u**n/n and at u=1/2 order 38 still leaves "
                       "6.4e-14. At 1/4 the same order is far past double."),
    "sqrt": CoreRange(0.25, 1.0, None, 0.625, "x = m * 4**k, m in [0.25, 1)"),
    "sinh": CoreRange(-1.0, 1.0, "odd", 0.0,
                      "own core; the exp identity cancels near zero"),
    "cosh": CoreRange(-1.0, 1.0, "even", 0.0,
                      "own core; keeps cosh(0) = 1 exact"),
    # Also narrowed on the map: 2.3e6 ulp on [-1,1], 0.76 at half that, in
    # sixteen coefficients instead of twenty-one that fail.
    "tanh": CoreRange(-0.5, 0.5, "odd", 0.0, "own core; exp form outside"),
    # 9.1e11 ulp on [-1,1]; 0.86 at half that.
    "asinh": CoreRange(-0.5, 0.5, "odd", 0.0, "own core; log identity outside"),
    "atanh": CoreRange(-0.5, 0.5, "odd", 0.0,
                       "own core; log1p identity outside"),
    # sinc is ENTIRE, so its series converges everywhere and the only
    # question is how many terms. A NARROW core is right on principle:
    # sin(x)/x needs a core only where it cancels, near zero -- beyond that
    # the quotient is well conditioned and the identity is better than any
    # polynomial. Measured: 9 coefficients at 0.51 ulp on [-1,1], against a
    # 256-coefficient polyspline at 11.3 on [-3,3].
    "sinc": CoreRange(-1.0, 1.0, "even", 0.0,
                      "sin(x)/x; the core covers only where it cancels"),
}

_REFERENCE_OVERRIDES: dict[str, Callable[[Any], Any]] = {}


def _reference(core: str, digits: int = 40) -> Callable[[Any], Any]:
    """The reference every core is scored against: the compiled program.

    This used to hand back an ``mpmath`` function, which made a library the
    authority on what these functions ARE -- the same borrowing this pack
    exists to end, moved from run time to bake time. It is now the SymPy
    derivation compiled to AbstractTensor Python and run at a longer order
    and more limbs, so there is one implementation and a disagreement always
    names a culprit.

    Scalar in, scalar out, for the node-sampling callers.
    """

    from .signal_symbolic import exact_evaluator

    spec = CORE_RANGES[str(core)]
    radius = max(abs(float(spec.low)), abs(float(spec.high)))
    return exact_evaluator(str(core), radius, digits=int(digits))


# --------------------------------------------------------------------------
# A baked core and its admission
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BakedCore:
    """One core's permanent numbers and the residual they actually achieve."""

    core: str
    family: str
    epsilon: float
    low: float
    high: float
    values: tuple[float, ...]
    #: The remainder of each value that a double could not hold. An exact
    #: core's coefficients are not representable, so keeping this second half
    #: lets the evaluation run in double-double and stop being the limiting
    #: error. Empty for families where the coefficients are a fit and their
    #: own residual dominates anything the evaluation could contribute.
    corrections: tuple[float, ...] = ()
    segments: int = 1
    centre: float = 0.0
    structure: str | None = None
    #: Worst RELATIVE residual measured against the reference on this core's
    #: own interval at bake time, and what ``admitted`` judges. Relative, not
    #: absolute: absolute minimax spreads error uniformly, but near a
    #: function's zero the values are tiny, so uniform absolute error is
    #: enormous relative error. Measured on sine, the absolute-fitted
    #: polyspline reached 6.66e-16 absolute and 9099 ULP at once.
    measured_error: float = float("nan")
    #: Kept for continuity with the absolute-targeted measurements, and
    #: because a growing function's absolute error is the honest one there.
    measured_absolute: float = float("nan")
    note: str = ""

    @property
    def admitted(self) -> bool:
        return bool(self.measured_error <= self.epsilon)

    @property
    def step(self) -> float:
        if self.family != "lut":
            raise AttributeError("only a lut core has a node step")
        return (self.high - self.low) / (len(self.values) - 1)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "core": self.core, "family": self.family, "epsilon": self.epsilon,
            "low": self.low, "high": self.high, "segments": self.segments,
            "structure": self.structure, "constants": len(self.values),
            "measured_error": self.measured_error,
            "measured_absolute": self.measured_absolute,
            "admitted": self.admitted,
            "note": self.note,
        }


#: Digits a single float64 limb carries.
DIGITS_PER_LIMB = 15.95


def working_digits(epsilon: float) -> int:
    """Derivation precision for a target, with guard digits.

    A CONSTANT here is the thing that makes a bake non-convergent: 60 digits
    is generous until the target asks for 70, and then no amount of growth can
    reach it. Guard digits are added rather than assumed, so the derivation is
    always meaningfully finer than what is being asked of it.
    """

    return int(max(30, math.ceil(-math.log10(epsilon)) + 25))


def limbs_for(epsilon: float) -> int:
    """How many float64 limbs a coefficient needs to hold this target.

    One limb is ~16 digits, so a target below 1e-16 is unreachable in a single
    double no matter how exact the derivation was -- the STORAGE truncates it.
    """

    from .extended_precision import SENSIBLE_LIMIT

    # Sized from the WORKING precision, not from the target. Sizing to the
    # target exactly makes the evaluation error the same size as the thing
    # being targeted -- at 1e-15 that is one limb, one limb is double, and the
    # core then measures ordinary double rounding no matter how exact its
    # coefficients are. ``working_digits`` already states how much finer than
    # the target the derivation must be, so the limb count follows from it
    # rather than from a floor chosen here.
    needed = int(math.ceil(working_digits(epsilon) / DIGITS_PER_LIMB))
    return int(min(max(needed, 1), SENSIBLE_LIMIT))


def validate_epsilon(epsilon: float | None) -> float:
    """Accept any target the representation can actually reach.

    The old floor of 1e-16 encoded an assumption that is no longer true: that
    a coefficient is one double and a result is one double. With limbs, the
    reachable floor is set by how many of them the target implies, so the
    check is against THAT rather than against a constant.
    """

    from .extended_precision import SENSIBLE_LIMIT

    epsilon = 1.0e-9 if epsilon is None else float(epsilon)
    floor = 10.0 ** -(SENSIBLE_LIMIT * DIGITS_PER_LIMB)
    if not math.isfinite(epsilon) or not floor <= epsilon <= 1.0e-2:
        raise ValueError(
            f"signal-math epsilon (relative) must be finite and in "
            f"[{floor:.0e}, 1e-2]; below that an expansion is the wrong "
            f"representation and a fixed-point significand is wanted"
        )
    return epsilon


def _horner(argument: Any, coefficients: Sequence[float]) -> Any:
    result = argument * 0.0 + float(coefficients[-1])
    for coefficient in reversed(tuple(coefficients)[:-1]):
        result = result * argument + float(coefficient)
    return result


def _index_tensor(value: Any) -> Any:
    for attribute in ("long", "int"):
        caster = getattr(value, attribute, None)
        if callable(caster):
            try:
                return caster()
            except Exception:
                continue
    return AbstractTensor.get_tensor(
        np.asarray(value.tolist(), dtype=np.int64), like=value,
    )


def _evaluate_extended(reduced: Any, core: BakedCore,
                       collapse: bool = True) -> Any:
    """Horner in double-double over coefficients that carry their remainder.

    An exact core's coefficients are the function's own Taylor coefficients,
    and the limiting error is then the EVALUATION, not the approximation: a
    Horner chain in double commits a rounding at every one of its twenty
    steps. Measured on the sine core, the identical series goes from 82% of
    results correctly rounded to 100%, ahead of libm's 97.7%, with no change
    to the coefficients at all.
    """

    from . import extended_precision as xp

    values, corrections = core.values, core.corrections
    # The width is read off the coefficients, not fixed at two: a core baked
    # for a finer target carries more limbs and must be EVALUATED at that
    # width or the storage was pointless.
    width = 1 + len(corrections[0])
    with xp.precision(width):
        if core.family == "series":
            base = reduced - core.centre
        elif core.structure == "factored":
            base = reduced
        else:
            base = reduced * reduced
        total = xp.constant_limbs(reduced, (values[-1],) + tuple(corrections[-1]))
        for index in range(len(values) - 2, -1, -1):
            total = total * base + xp.constant_limbs(
                reduced, (values[index],) + tuple(corrections[index]))
        if core.family != "series" and core.structure in ("factored", "odd"):
            total = total * reduced
    return xp.collapse(total) if collapse else total


def evaluate_core(reduced: Any, core: BakedCore) -> Any:
    """Evaluate a baked core on an argument ALREADY inside its interval."""

    # ``exact`` is the same FORM as ``structured`` -- the parity is carried by
    # the expression either way -- and differs only in where the coefficients
    # came from, so it evaluates identically.
    if core.family in ("structured", "exact"):
        if core.corrections:
            return _evaluate_extended(reduced, core)
        if core.structure == "factored":
            # f(u) = u * P(u): the root is exact, so the polynomial never has
            # to fit across a zero and relative accuracy stays attainable.
            return reduced * _horner(reduced, core.values)
        square = reduced * reduced
        polynomial = _horner(square, core.values)
        return reduced * polynomial if core.structure == "odd" else polynomial

    if core.family == "series":
        if core.corrections:
            return _evaluate_extended(reduced, core)
        return _horner(reduced - core.centre, core.values)

    if core.family == "polyspline":
        width = len(core.values) // core.segments
        span = (core.high - core.low) / core.segments
        half = 0.5 * span

        def segment(index: int) -> Any:
            centre = core.low + span * (index + 0.5)
            block = core.values[index * width:(index + 1) * width]
            return _horner((reduced - centre) * (1.0 / half), block)

        result = segment(core.segments - 1)
        for index in range(core.segments - 2, -1, -1):
            result = _where(
                reduced < core.low + span * (index + 1), segment(index), result,
            )
        return result

    highest = len(core.values) - 2
    table = AbstractTensor.get_tensor(
        np.asarray(core.values, dtype=np.float64), like=reduced,
    )
    position = (reduced - core.low) * (1.0 / core.step)
    # Clamp the BASE INDEX, never the position: pinning the position pins the
    # interpolation weight to zero at the top endpoint, degrading the table to
    # nearest-node lookup there (measured h/2 error). Sine hid this
    # completely, because sin'(pi/2) = 0 makes that one endpoint free, and
    # sine was the only function the old LUT path ever tested.
    base = position.floor()
    base = _where(base < 0.0, base * 0.0, base)
    base = _where(
        base > float(highest), base * 0.0 + float(highest), base,
    )
    weight = position - base
    index = _index_tensor(base)
    left = table.index_select(0, index)
    right = table.index_select(0, index + 1)
    if not core.corrections:
        return left + (right - left) * weight

    # With node corrections the stored values stop contributing error at all,
    # and the interpolation runs in double-double so the subtraction of two
    # neighbouring nodes -- which cancels, and cancellation is where a table
    # actually loses digits -- is exact. What remains is the LINEAR
    # TRUNCATION between nodes, which is a property of the spacing and not of
    # the arithmetic, so this cannot rescue a table that is simply too coarse.
    from . import extended_precision as xp

    low_table = AbstractTensor.get_tensor(
        np.asarray(core.corrections, dtype=np.float64), like=reduced,
    )
    # Both lookups happen OUTSIDE the precision block. Index arithmetic
    # inside it would be promoted by the shim like any other operand, and
    # ``index + 1`` would arrive at ``index_select`` as a float expansion.
    near_low = low_table.index_select(0, index)
    far_low = low_table.index_select(0, index + 1)
    with xp.precision(2):
        near = xp.pair(left, near_low)
        far = xp.pair(right, far_low)
        result = near + (far - near) * weight
    return xp.collapse(result)


def _measure(core: BakedCore) -> BakedCore:
    """Score a baked core against the same program run wider.

    The difference is formed in EXTENDED arithmetic before it is collapsed.
    That matters more than it sounds: subtracting two nearly-equal doubles
    destroys exactly the digits being measured, so a difference computed in
    double puts a floor of one ulp under every measurement and a core baked
    for 1e-25 could never be scored as reaching it -- growth would stop at the
    floor of the INSTRUMENT rather than at the target.
    """

    from fractions import Fraction

    from . import extended_precision as xp

    digits = working_digits(core.epsilon)
    truth = _reference(core.core, digits=digits)

    positions = np.linspace(core.low, core.high, 801)
    tensor = AbstractTensor.get_tensor(positions)
    if core.corrections and core.family in ("structured", "exact", "series"):
        pieces = _evaluate_extended(tensor, core, collapse=False)
        limbs = [pieces] + list(getattr(pieces, "_limbs", ()))
    else:
        limbs = [evaluate_core(tensor, core)]
    columns = [np.asarray(limb.tolist(), dtype=np.float64).ravel()
               for limb in limbs]

    magnitude = np.empty(positions.size)
    absolute = np.empty(positions.size)
    for index, item in enumerate(positions):
        produced = sum((Fraction(float(column[index])) for column in columns),
                       Fraction(0))
        expected = truth(float(item))
        magnitude[index] = abs(float(expected))
        absolute[index] = abs(float(produced - expected))
    # A genuine zero of the function contributes its absolute error; dividing
    # there would report inf for a core that is exactly right.
    relative = np.where(magnitude > 0.0,
                        absolute / np.where(magnitude > 0.0, magnitude, 1.0),
                        absolute)
    return replace(
        core,
        measured_error=float(np.max(relative)),
        measured_absolute=float(np.max(absolute)),
    )


# --------------------------------------------------------------------------
# Fitting: every family grown to a measured target
# --------------------------------------------------------------------------


GROWTH_LIMITS: Mapping[str, int] = {
    "exact": 64, "series": 64, "structured": 20, "polyspline": 32, "lut": 1 << 22,
}


def _grow(build: Callable[[int], BakedCore],
          sizes: Sequence[int]) -> BakedCore:
    """First core in the ladder whose MEASURED residual meets its target."""

    best: BakedCore | None = None
    for size in sizes:
        core = build(int(size))
        if best is None or core.measured_error < best.measured_error:
            best = core
        if core.admitted:
            return core
    if best is None:
        raise ValueError("growth ladder was empty")
    return best


def _chebyshev_nodes(low: float, high: float, count: int) -> np.ndarray:
    index = np.arange(int(count))
    return 0.5 * (high - low) * (
        np.cos(np.pi * (index + 0.5) / int(count))[::-1] + 1.0
    ) + low


def _split_high_low(values, limbs: int = 2):
    """Each value as ``limbs`` doubles that sum to it.

    ``Fraction`` is exact and ``float(Fraction)`` is correctly rounded, so
    each step takes the nearest double and carries the remainder forward
    without loss. That is the whole reason no arbitrary-precision library is
    needed at bake time: these coefficients were rational all along.
    """

    from fractions import Fraction

    heads, tails = [], []
    for value in values:
        parts = []
        rest = value if isinstance(value, Fraction) else Fraction(value)
        for _ in range(max(int(limbs), 1)):
            head = float(rest)
            parts.append(head)
            rest = rest - Fraction(head)
        heads.append(parts[0])
        tails.append(tuple(parts[1:]))
    return tuple(heads), tuple(tails)


def _exact_structured_coefficients(core: str, structure: str, order: int,
                                   epsilon: float = 1.0e-15):
    """Structured coefficients, derived symbolically and split into limbs.

    This used to call ``mpmath.taylor`` on a callable, which DIFFERENTIATES
    NUMERICALLY: its error grows with order however much working precision it
    is given, so the bake plateaued at 6.29e-18 and growth could not pass the
    plateau because longer series scored worse. The coefficients now come from
    ``signal_symbolic``, where SymPy derives them from the function's own
    identity as exact rationals, and the only thing left to decide is how many
    limbs to keep them in.
    """

    from .signal_symbolic import (
        limb_decomposition, order_to_degree, structured_coefficients,
    )

    count = max(int(order) // 2 + 1, 2)
    coefficients = structured_coefficients(
        str(core), order_to_degree(str(core), count),
    )
    limbs = limbs_for(epsilon)
    split = [limb_decomposition(value, limbs) for value in coefficients]
    return (tuple(parts[0] for parts in split),
            tuple(tuple(parts[1:]) for parts in split))


def _structured_coefficients(core: str, structure: str, low: float,
                             high: float, degree: int) -> tuple[float, ...]:
    """Near-minimax coefficients in ``u = y*y``; the square carries the parity.

    Fitted in ``u`` DIRECTLY. Interpolating in the Chebyshev variable and then
    evaluating in ``u`` is a basis mismatch that leaves parity exact and the
    values useless -- measured 3.3e-01 on sine, which looks like success on
    precisely the metrics one checks first.
    """

    from .signal_symbolic import structured_coefficients

    reference = _reference(core)
    # The value at the origin is not a limit to be probed -- it is the
    # structured series' leading coefficient, known exactly. Taking a
    # numerical derivative or sampling at 1e-25 to find it was estimating
    # something already derived.
    origin = float(_split_high_low(
        [structured_coefficients(str(core), 6)[0]], 1)[0][0])

    if structure == "factored":
        def fitted(u: float) -> float:
            return origin if u == 0.0 else float(reference(u)) / u
    elif structure == "odd":
        def fitted(u: float) -> float:
            if u <= 0.0:
                return origin
            root = math.sqrt(u)
            return float(reference(root)) / root
    else:
        def fitted(u: float) -> float:
            return origin if u <= 0.0 else float(reference(math.sqrt(u)))

    # A factored core fits in the ORIGINAL argument over the core's own
    # interval; odd and even cores fit in its square, which is what makes
    # the parity structural.
    count = max(8 * (int(degree) + 1), 64)
    if structure == "factored":
        nodes = _chebyshev_nodes(float(low), float(high), count)
    else:
        reach = max(abs(float(low)), abs(float(high))) ** 2
        nodes = _chebyshev_nodes(0.0, reach, count)
    samples = np.asarray(
        [fitted(float(node)) for node in nodes], dtype=np.float64,
    )
    # Weight by 1/|g| so the fit minimises RELATIVE residual. For a factored
    # or odd core the factor outside the polynomial is exact, so g's relative
    # error IS f's relative error -- this weighting, not a larger degree, is
    # what buys reference accuracy.
    #
    # Fitted in the CHEBYSHEV basis and converted to powers afterwards. A
    # power-basis least squares is rank-deficient at the degrees atan and
    # asinh need (numpy raised RankWarning and the fits lost to a 256-constant
    # polyspline). The conversion is exact rescaling, not a second fit.
    weights = 1.0 / np.maximum(np.abs(samples), np.finfo(np.float64).tiny)
    low, high = float(nodes[0]), float(nodes[-1])
    chebyshev = np.polynomial.chebyshev.Chebyshev(
        np.polynomial.chebyshev.chebfit(
            2.0 * (nodes - low) / (high - low) - 1.0,
            samples, int(degree), w=weights,
        ),
        domain=[low, high], window=[-1.0, 1.0],
    )
    coefficients = list(
        chebyshev.convert(
            kind=np.polynomial.Polynomial, domain=[low, high],
            window=[low, high],
        ).coef
    )
    coefficients += [0.0] * (int(degree) + 1 - len(coefficients))
    if structure in ("factored", "odd"):
        # Pin P(0) to the exact limit. A weighted least squares spreads its
        # residual and leaves the constant term off by ~4e-13, which IS the
        # relative error as the argument approaches zero -- the one place the
        # factored form was supposed to make exact. Measured on log1p: the
        # worst relative error sat at x = 1.25e-04 with the value itself at
        # 1.25e-04, i.e. entirely the constant term.
        coefficients[0] = origin
    return tuple(float(value) for value in coefficients)


def fit_structured(core: str, epsilon: float | None = None) -> BakedCore:
    """Grow a parity-structured core's degree until it measures under epsilon."""

    epsilon = validate_epsilon(epsilon)
    spec = CORE_RANGES[core]
    if spec.structure is None:
        raise ValueError(f"{core!r} declares no parity; fit it as a plain core")

    def build(degree: int) -> BakedCore:
        return _measure(BakedCore(
            core=core, family="structured", epsilon=epsilon,
            low=spec.low, high=spec.high, structure=spec.structure,
            values=_structured_coefficients(
                core, spec.structure, spec.low, spec.high, degree,
            ),
            note=f"{spec.note}; {spec.structure} in y*y, degree {degree}",
        ))

    return _grow(build, range(2, GROWTH_LIMITS["structured"] + 1, 2))


def fit_exact(core: str, epsilon: float | None = None) -> BakedCore:
    """Grow a parity-structured core of EXACT coefficients until it measures.

    The accuracy family. Same structural guarantees as ``structured`` -- the
    parity is in the form, so ``sin(0)`` is exactly zero and there are no
    knots -- with the coefficients taken rather than fitted, which is what
    lets it reach the working type's own precision instead of a fit's floor.
    """

    epsilon = validate_epsilon(epsilon)
    spec = CORE_RANGES[core]
    if spec.structure is None:
        raise ValueError(f"{core!r} declares no parity; fit it as a plain core")
    if spec.structure == "factored":
        # A root-factored core has an exact form of its own: f(u) = u * P(u)
        # with P's coefficients taken straight from the Taylor series, the
        # exactly-zero constant term DROPPED rather than carried. Routing this
        # to the plain series (as this function used to) threw the factored
        # evaluator away, which is the whole reason the root is exact.
        def build_factored(order: int) -> BakedCore:
            from .signal_symbolic import structured_coefficients

            coefficients = structured_coefficients(str(core), int(order))
            highs, lows = _split_high_low(coefficients, limbs_for(epsilon))
            return _measure(BakedCore(
                core=core, family="exact", epsilon=epsilon,
                low=spec.low, high=spec.high, structure="factored",
                values=highs, corrections=lows,
                note=f"{spec.note}; exact factored series, order {order}",
            ))

        return _grow(build_factored,
                     range(3, GROWTH_LIMITS["exact"] + 1, 2))

    def build(order: int) -> BakedCore:
        _exact_coefficients = _exact_structured_coefficients(
            core, spec.structure, order, epsilon)
        return _measure(BakedCore(
            core=core, family="exact", epsilon=epsilon,
            low=spec.low, high=spec.high, structure=spec.structure,
            values=_exact_coefficients[0], corrections=_exact_coefficients[1],
            note=f"{spec.note}; exact {spec.structure} series, order {order}",
        ))

    return _grow(build, range(3, GROWTH_LIMITS["series"] + 1, 2))


def _chebyshev_segment(function: Callable[[float], float], low: float,
                       high: float, degree: int) -> tuple[float, ...]:
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    chebyshev = np.polynomial.chebyshev.Chebyshev.interpolate(
        lambda t: np.asarray([
            float(function(centre + half * float(item)))
            for item in np.atleast_1d(t)
        ]),
        int(degree), domain=[-1.0, 1.0],
    )
    power = chebyshev.convert(kind=np.polynomial.Polynomial)
    coefficients = list(power.coef) + [0.0] * (int(degree) + 1 - len(power.coef))
    return tuple(float(value) for value in coefficients)


def fit_polyspline(core: str, epsilon: float | None = None,
                   *, degree: int = 7) -> BakedCore:
    """Grow the segment count until the fitted spline measures under epsilon."""

    epsilon = validate_epsilon(epsilon)
    spec = CORE_RANGES[core]
    reference = _reference(core)

    def build(segments: int) -> BakedCore:
        edges = np.linspace(spec.low, spec.high, segments + 1)
        flattened: list[float] = []
        for index in range(segments):
            flattened.extend(_chebyshev_segment(
                reference, float(edges[index]), float(edges[index + 1]), degree,
            ))
        return _measure(BakedCore(
            core=core, family="polyspline", epsilon=epsilon,
            low=spec.low, high=spec.high, values=tuple(flattened),
            segments=segments,
            note=f"{spec.note}; {segments} segments of degree {degree}",
        ))

    ladder, segments = [], 1
    while segments <= GROWTH_LIMITS["polyspline"]:
        ladder.append(segments)
        segments *= 2
    return _grow(build, ladder)


def fit_series(core: str, epsilon: float | None = None) -> BakedCore:
    """A plain series core, coefficients derived rather than differentiated.

    The ``series`` family exists as the DIFFERENTIAL set: the same numbers
    without the parity carried in the form, so the structured form's benefit
    can be measured rather than asserted.
    """

    from .signal_symbolic import (
        limb_decomposition, order_for, reduced_series, TRANSCENDENTALS,
    )
    import sympy

    epsilon = validate_epsilon(epsilon)
    spec = CORE_RANGES[core]
    radius = max(abs(float(spec.low) - float(spec.centre)),
                 abs(float(spec.high) - float(spec.centre)))

    def build(order: int) -> BakedCore:
        polynomial = sympy.Poly(
            reduced_series(str(core), int(order)), sympy.Symbol("z"),
        )
        exact = [polynomial.coeff_monomial(sympy.Symbol("z") ** power)
                 if power else polynomial.coeff_monomial(1)
                 for power in range(int(order) + 1)]
        limbs = limbs_for(epsilon)
        split = [limb_decomposition(value, limbs) for value in exact]
        return _measure(BakedCore(
            core=core, family="series", epsilon=epsilon,
            low=spec.low, high=spec.high, centre=spec.centre,
            values=tuple(parts[0] for parts in split),
            corrections=tuple(tuple(parts[1:]) for parts in split),
            note=f"{spec.note}; order {order} about {spec.centre:g}, derived",
        ))

    try:
        count = order_for(str(core), radius, digits=int(-math.log10(epsilon)))
    except (KeyError, ValueError):
        count = 24
    return _grow(build, range(3, max(2 * count + 2, 12), 2))


@dataclass(frozen=True)
class AnglePalette:
    """Exact values for an angle set the program is KNOWN to use.

    Every other family approximates a function over an interval, because the
    argument is not known until it arrives. When the argument set IS known --
    a DFT or CQT twiddle set, a fixed sweep, a quantised control surface --
    there is nothing left to approximate: store the correctly-rounded value
    of each angle the program will actually ask for.

    Measured against a 60-digit reference on the turn lattice ``k/N``::

        N       palette          computed as sin(2*pi*k/N)
        1024    0.00 ulp p95     11.00 ulp p95
        4096    0.00 ulp p95     11.00 ulp p95

    Zero, and better than libm, for a reason worth stating: the usual route
    forms ``2*pi*k/N`` first, which is not the exact angle, and then computes
    an accurate function of a slightly wrong argument. A palette never forms
    that intermediate at all.

    Two constraints, both load-bearing:

    * Lookup is BY INDEX, never by searching for a value. A caller who knows
      their angle set already holds the index -- it is their loop variable.
    * The palette is admitted for its declared set and MUST refuse anything
      else. Serving a nearby angle would silently return a different
      function's answer, which is the defect class the shape guards exist to
      end.

    Footprint is ``16*N`` bytes for the sine/cosine pair: 16 KiB at N=1024,
    64 KiB at N=4096.
    """

    #: How many equal divisions of one turn. Index ``k`` means angle ``k/N``.
    divisions: int
    sine: tuple[float, ...]
    cosine: tuple[float, ...]
    #: The remainder of each entry that a double could not hold. A palette is
    #: baked once and read many times, so the second half costs one more array
    #: at bake time and nothing at all per lookup -- and it lets a caller that
    #: wants more than double (a long DFT accumulating twiddle by twiddle) have
    #: it, rather than being capped by the storage format.
    sine_low: tuple[float, ...] = ()
    cosine_low: tuple[float, ...] = ()
    measured_error: float = float("nan")

    @property
    def admitted(self) -> bool:
        # Correctly rounded, which is half an ulp -- not exactly zero. The
        # placed cardinal values differ from the reference by the reference's
        # OWN residue for pi, so an equality test rejects the entries that are
        # more right than what they are being compared to. A palette that
        # cannot claim correct rounding is a table with extra steps.
        return self.measured_error <= 0.5

    def to_mapping(self) -> dict[str, Any]:
        return {
            "family": "palette", "divisions": self.divisions,
            "entries": len(self.sine), "bytes": 16 * len(self.sine),
            "measured_error": self.measured_error,
            "admitted": self.admitted,
        }


def bake_angle_palette(divisions: int) -> AnglePalette:
    """Bake the exact turn lattice ``k/divisions`` for one known angle set.

    Only the FIRST QUADRANT is computed; the rest is index arithmetic and a
    sign. That is not a storage trick bolted on -- it is what makes the
    symmetries exact rather than merely accurate. Evaluating ``sin`` at the
    half turn is PLACED exactly, while a computed reference returns its own
    residue for pi at whatever width it was derived to, not
    zero, and ``s[k] + s[N-k]`` misses zero by the same amount. Folding makes
    ``sin`` at 0 and the half turn exactly +0, the quarter turns exactly +-1,
    and odd symmetry hold to the bit, because those values are placed rather
    than computed.

    It also costs a quarter of the memory: one quadrant of sine serves both
    sine and cosine, since ``cos(k) = sin(k + N/4)``.
    """


    divisions = int(divisions)
    if divisions < 1:
        raise ValueError(f"a palette needs at least one division, got {divisions}")
    if divisions % 4:
        raise ValueError(
            f"a palette divides the turn into quadrants; {divisions} is not a "
            f"multiple of four"
        )
    from fractions import Fraction

    from .signal_symbolic import constant_rational, exact_evaluator

    quarter = divisions // 4
    digits = 40
    turn = 2 * constant_rational("pi", digits) / divisions
    sine_of = exact_evaluator("sin", float(turn * quarter) + 0.1, digits=digits)
    # One quadrant, endpoints placed exactly rather than computed.
    exact_quadrant = ([Fraction(0)]
                      + [sine_of(turn * index) for index in range(1, quarter)]
                      + [Fraction(1)])
    quadrant = [float(value) for value in exact_quadrant]
    quadrant_low = [
        float(value - Fraction(float(value))) for value in exact_quadrant
    ]

    def fold(table: list, index: int) -> float:
        """Quadrant folding. The same reflections and sign apply to both
        halves of an entry, so the correction stays attached to its value."""

        index %= divisions
        if index <= quarter:
            return table[index]
        if index <= 2 * quarter:
            return table[2 * quarter - index]
        return -fold(table, index - 2 * quarter)

    def exact_quadrant_value(index: int) -> Fraction:
        """The folded EXACT entry, for scoring against what was stored."""

        index %= divisions
        if index <= quarter:
            return exact_quadrant[index]
        if index <= 2 * quarter:
            return exact_quadrant[2 * quarter - index]
        return -exact_quadrant_value(index - 2 * quarter)

    sine = tuple(fold(quadrant, index) for index in range(divisions))
    cosine = tuple(fold(quadrant, index + quarter) for index in range(divisions))
    sine_low = tuple(fold(quadrant_low, index) for index in range(divisions))
    cosine_low = tuple(
        fold(quadrant_low, index + quarter) for index in range(divisions)
    )
    if True:
        # Scored against the same exact values the table was
        # rounded from, so "0" means correctly rounded rather than
        # self-consistent.
        # Scored in ULP OF FULL SCALE, not of each value. Sine and cosine are
        # bounded by one, so near a zero the relative measure is meaningless
        # and actively misleading: this palette PLACES an exact zero at the
        # half turn while the reference returns its own residue for pi
        # (~1e-61), and a relative score calls the exact answer wrong by 1e15
        # ulp. Absolute error against the function's range is the honest
        # unit for a bounded function.
        scale = float(np.spacing(1.0))
        worst = 0.0
        for index in range(divisions):
            for stored, exact in (
                (sine[index], exact_quadrant_value(index)),
                (cosine[index], exact_quadrant_value(index + quarter)),
            ):
                worst = max(worst, abs(stored - float(exact)) / scale)
    return AnglePalette(divisions, sine, cosine, sine_low, cosine_low,
                        measured_error=worst)


def fit_lut(core: str, epsilon: float | None = None,
            *, generator: BakedCore | None = None) -> BakedCore:
    """Grow a node table until it measures under epsilon.

    The nodes are a PRERUN of the series core: the series computes the values
    and the table caches them at whatever resolution linear interpolation
    needs to hold the same target. One authority for what the function is; the
    table is only a decision about how to pay for it. Where the series cannot
    meet the target -- ``atan`` on its reduced interval, measured -- the
    arbitrary-precision reference generates the nodes and the note says so
    rather than quietly inheriting an unadmitted generator's error.
    """

    epsilon = validate_epsilon(epsilon)
    spec = CORE_RANGES[core]
    if generator is None:
        generator = fit_series(core, epsilon)
    if generator.admitted:
        source = "series prerun"

        def evaluate(positions: np.ndarray) -> np.ndarray:
            return np.asarray(
                evaluate_core(
                    AbstractTensor.get_tensor(positions), generator,
                ).tolist(), dtype=np.float64,
            ).ravel()

        def evaluate_low(positions: np.ndarray) -> np.ndarray:
            """The half of each node a double cannot hold.

            Taken from the generator itself rather than from arbitrary
            precision: the series core already carries its coefficients'
            corrections, so evaluating it extended and keeping BOTH limbs
            costs one tensor pass over the nodes instead of one mpmath call
            per node -- which matters at a table size that runs to millions.
            """

            if not generator.corrections:
                return np.zeros_like(positions)
            extended = _evaluate_extended(
                AbstractTensor.get_tensor(positions), generator, collapse=False,
            )
            limbs = getattr(extended, "_limbs", ())
            if not limbs:
                return np.zeros_like(positions)
            return np.asarray(limbs[0].tolist(), dtype=np.float64).ravel()
    else:
        source = (f"reference prerun (series fell short at "
                  f"{generator.measured_error:.2e})")
        reference = _reference(core)

        def evaluate(positions: np.ndarray) -> np.ndarray:
            return np.asarray(
                [float(reference(float(item))) for item in positions],
                dtype=np.float64,
            )

        def evaluate_low(positions: np.ndarray) -> np.ndarray:
            from fractions import Fraction

            from .signal_symbolic import exact_evaluator

            radius = max(abs(float(spec.low)), abs(float(spec.high)))
            oracle = exact_evaluator(str(core), radius, digits=40)
            pieces = []
            for item in positions:
                value = oracle(float(item))
                pieces.append(float(value - Fraction(float(value))))
            return np.asarray(pieces, dtype=np.float64)

    def build(intervals: int) -> BakedCore:
        positions = np.linspace(spec.low, spec.high, intervals + 1)
        return _measure(BakedCore(
            core=core, family="lut", epsilon=epsilon,
            low=spec.low, high=spec.high,
            values=tuple(float(value) for value in evaluate(positions)),
            corrections=tuple(
                float(value) for value in evaluate_low(positions)
            ),
            note=f"{spec.note}; {intervals} intervals from {source}",
        ))

    ladder, intervals = [], 16
    while intervals <= GROWTH_LIMITS["lut"]:
        ladder.append(intervals)
        intervals *= 2
    return _grow(build, ladder)


#: Families the ``best`` selector will try. ``lut`` is excluded on measured
#: evidence, not taste: it needed 4.2 million constants to reach 6235 ulp on
#: sine, so it can never win a selection made on accuracy per constant. Ask
#: for it by name when uniform per-call cost is the reason.
SELECTABLE_FAMILIES = ("exact", "structured", "series", "polyspline")


def fit_best(core: str, epsilon: float | None = None) -> "BakedCore":
    """Fit every selectable family and keep the one that MEASURES best.

    One family per core set was the wrong granularity. Measured on sine at
    1e-15, ``series`` reached 1.0 ulp in 16 constants while ``structured``
    reached 4.7 in 11 and ``polyspline`` only 372 in 256 -- and on ``atan``
    the ranking inverts, because no order of ``atan``'s series admits at all.
    Selecting per core costs one extra bake and takes the better of both.

    Ties on measured error go to the smaller core, so a family only earns
    extra constants by being genuinely more accurate.
    """

    epsilon = validate_epsilon(epsilon)
    candidates = []
    for family in SELECTABLE_FAMILIES:
        try:
            candidates.append(fit_core(core, family, epsilon))
        except Exception:
            continue
    if not candidates:
        raise ValueError(f"no family could fit core {core!r}")
    return min(candidates, key=lambda c: (c.measured_error, len(c.values)))


def fit_core(core: str, family: str = DEFAULT_FAMILY,
             epsilon: float | None = None, **options: Any) -> BakedCore:
    """Bake one core of one family, sized by measurement against epsilon.

    ``structured`` falls back to ``polyspline`` for a core with no declared
    parity, so asking for the best available form does not require the caller
    to know which functions happen to be odd or even.
    """

    family = str(family)
    structure = CORE_RANGES[core].structure
    if family == "exact" and structure is None:
        # No parity to carry structurally, so the exact form is the plain
        # series -- NOT polyspline, which is a fit and therefore has a floor.
        # Measured: exp reaches 1.0 ulp as a series and 3.1 as a polyspline,
        # log 1.4 against 3317.
        family = "series"
    if family == "structured" and structure is None:
        # A core with no parity has no structure for the "structured" form to
        # carry, so the choice is between a FIT and the function's own series.
        # This used to pick the fit, from a time when the series coefficients
        # were themselves fitted; they are derived exactly now, so the fit's
        # residual floor is a floor for no reason -- measured three lines
        # above, exp is 1.0 ulp as a series against 3.1 as a polyspline and
        # log 1.4 against 3317. The kernels cannot emit a segment-selection
        # chain either, so the fit was also unbakeable.
        #
        # Conditional, because not every core HAS a series: sqrt is reached by
        # Newton, which is a fixed point of the answer rather than a Taylor
        # expansion, and there is nothing for the identity table to derive.
        # Where no series exists the fit is the only form, and the kernel
        # generator refusing to emit it is the honest outcome.
        from .signal_symbolic import TRANSCENDENTALS

        family = "series" if str(core) in TRANSCENDENTALS else "polyspline"
    fitters = {
        "exact": fit_exact, "structured": fit_structured,
        "polyspline": fit_polyspline, "series": fit_series, "lut": fit_lut,
    }
    try:
        fitter = fitters[family]
    except KeyError as error:
        raise ValueError(
            f"family must be one of {FAMILIES!r}, got {family!r}"
        ) from error
    return fitter(core, epsilon, **options)


# --------------------------------------------------------------------------
# Prebake parameter sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrebakeSettings:
    """One named point in (mode, family, epsilon) with the cores it needs."""

    name: str
    mode: str
    family: str
    epsilon: float
    note: str

    def cores(self) -> tuple[str, ...]:
        return PRIMITIVES if self.mode == "implied" else tuple(CORE_RANGES)


#: The dispatcher resolves a request to one of these. They exist so a caller
#: asks for a QUALITY, not for a degree and a segment count.
PREBAKE_SETS: Mapping[str, PrebakeSettings] = {
    "draft": PrebakeSettings(
        "draft", "implied", "structured", 1.0e-6,
        "smallest and loosest; identities off six primitive cores",
    ),
    "audio": PrebakeSettings(
        "audio", "direct", "structured", 1.0e-9,
        "well past 24-bit; every method on its own core",
    ),
    "double": PrebakeSettings(
        "double", "direct", "best", 1.0e-15,
        "everything the working type can hold; the dispatch default",
    ),
    "reference": PrebakeSettings(
        "reference", "direct", "best", 1.0e-15,
        "per-core best measured family; the default for analysis work",
    ),
    "table": PrebakeSettings(
        "table", "direct", "lut", 1.0e-9,
        "uniform per-call cost; large constants, no polynomial chain",
    ),
    "definitional": PrebakeSettings(
        "definitional", "direct", "series", 1.0e-9,
        "the series itself, for differential checks against the rest",
    ),
}


@dataclass(frozen=True)
class CoreSet:
    """Every core one prebake setting needs, baked and scored together."""

    settings: PrebakeSettings
    cores: Mapping[str, BakedCore]

    def __getitem__(self, name: str) -> BakedCore:
        try:
            return self.cores[str(name)]
        except KeyError as error:
            raise KeyError(
                f"{self.settings.name!r} holds no core {name!r}; it bakes "
                f"{tuple(self.cores)!r}"
            ) from error

    @property
    def admitted(self) -> bool:
        return all(core.admitted for core in self.cores.values())

    @property
    def constants(self) -> int:
        return sum(len(core.values) for core in self.cores.values())

    def shortfalls(self) -> tuple[BakedCore, ...]:
        return tuple(
            core for core in self.cores.values() if not core.admitted
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                "name": self.settings.name, "mode": self.settings.mode,
                "family": self.settings.family, "epsilon": self.settings.epsilon,
                "note": self.settings.note,
            },
            "admitted": self.admitted, "constants": self.constants,
            "cores": {
                name: core.to_mapping() for name, core in self.cores.items()
            },
        }


_PREBAKED: dict[tuple[str, str, str, float], CoreSet] = {}


def prebake(name: str = "reference") -> CoreSet:
    """Bake (once per process) every core one named setting needs."""

    try:
        settings = PREBAKE_SETS[str(name)]
    except KeyError as error:
        raise KeyError(
            f"unknown prebake set {name!r}; expected one of "
            f"{tuple(PREBAKE_SETS)!r}"
        ) from error
    key = (settings.name, settings.mode, settings.family, settings.epsilon)
    existing = _PREBAKED.get(key)
    if existing is not None:
        return existing
    baked = CoreSet(settings, {
        core: (
            fit_best(core, settings.epsilon) if settings.family == "best"
            else fit_core(core, settings.family, settings.epsilon)
        )
        for core in settings.cores()
    })
    _PREBAKED[key] = baked
    return baked


# --------------------------------------------------------------------------
# Reductions: the argument-mapping half of every method
# --------------------------------------------------------------------------


def _where(condition: Any, if_true: Any, if_false: Any) -> Any:
    """Select branchlessly: ``if_false + mask * (if_true - if_false)``.

    Not a workaround dressed up as a technique -- masked arithmetic is the
    normal way to write a selection inside a numeric core. It has no branch to
    mispredict, it vectorises, and every lane does the same work regardless of
    the data. Both branches are evaluated either way here: a tensor ``where``
    is not lazy, so the callers below already substitute a safe value into the
    inactive branch before any risky operation, and that discipline is what
    keeps ``0 * inf`` from appearing.

    It also matches the differential exactly. The mask is a step function, so
    it carries no gradient of its own -- measured: the backward of a bare
    ``(x > 0) * 1.0`` reports no gradient path at all. The upstream gradient
    therefore reaches ``if_true`` weighted by the mask and ``if_false`` by its
    complement, and the condition receives none. That is precisely what the
    authored ``where`` rule states: ``ga = g * indicator(cond)``,
    ``gb = g * (1 - indicator(cond))``, ``None`` for the condition.

    The practical reason to prefer it: the compiled REVERSE of a graph
    containing a tensor ``where`` currently stops in region planning, while
    this form compiles. ``AbstractTensor.where`` also has a second trap --
    it bypasses the backend's source-producing override and falls into the
    Python valuewise path, which calls ``tolist()`` and is fatal under SSA
    capture. Both are avoided by never emitting the operation.
    """

    mask = condition * 1.0
    return if_false + mask * (if_true - if_false)


def _tensor(value: Any) -> Any:
    return AbstractTensor.get_tensor(value)


def _magnitude(value: Any) -> Any:
    return _where(value < 0.0, -value, value)


def _sign(value: Any) -> Any:
    return _where(value < 0.0, value * 0.0 - 1.0, value * 0.0 + 1.0)


def _fold_quadrant(radians: Any) -> Any:
    """Reflect [-pi, pi] into [-pi/2, pi/2], preserving the sine."""

    upper = _where(
        radians > HALF_PI, (radians * 0.0 + math.pi) - radians, radians,
    )
    return _where(
        upper < -HALF_PI, (upper * 0.0 - math.pi) - upper, upper,
    )


def _turn_radians(turns: Any) -> Any:
    """Turn count to radians in [-pi, pi), reduced EXACTLY at any magnitude."""

    turns = _tensor(turns)
    fraction = turns - turns.floor()
    centred = _where(fraction > 0.5, fraction - 1.0, fraction)
    return centred * TAU


def _binade(value: Any) -> Any:
    """Nearest power of two, as a float tensor.

    Computed with AbstractTensor operators so it CAPTURES. An earlier version
    read the tensor out with ``tolist()`` and used NumPy, which materialises
    and cuts the graph -- that, not the libm dependency, is what blocks
    compilation. Deferring the exponent to the platform's ``log`` is fine for
    lowering (it becomes ``unary_double``, exactly as the existing trig pack
    already compiles), but it does mean this one step is not independent of
    the library we are replacing. A ``frexp`` primitive would remove even
    that; only the integer part of the result is ever used, so the exponent
    step needs to be right, not accurate.
    """

    return (value.log() * (1.0 / LN2) + 0.5).floor()


def _even_binade(value: Any) -> Any:
    """``k`` such that ``x / 4**k`` lands in [0.25, 1). Captures, as above."""

    # ceil via floor: this backend exposes only ``ceil_``, and -floor(-x)
    # is the portable spelling that lowers the same way everywhere.
    return -((value.log() * (-0.5 / LN2)).floor())


# --------------------------------------------------------------------------
# The surface
# --------------------------------------------------------------------------


class SignalMath:
    """The whole trigonometric surface over one baked core set.

    Every method is authored AbstractTensor arithmetic over baked constants,
    so the same definition captures for every backend. Methods come in two
    flavours where it matters: a ``_turns`` entry taking CYCLES, which is
    exact at any magnitude, and a radian entry, which is not and says so.
    """

    def __init__(self, cores: CoreSet):
        self.cores = cores

    # -- circular ---------------------------------------------------------

    def _octant(self, turns: Any) -> tuple[Any, Any]:
        """Reduce a turn count to ``|y| <= pi/4`` and its quadrant index.

        Every step is exact: ``t - floor(t)`` is exact at any magnitude, the
        quadrant index is an integer, and ``n/4`` is a dyadic rational so
        ``fraction - n/4`` subtracts without rounding. Only the final scale by
        TAU rounds, and it acts on a value already below pi/4.
        """

        turns = _tensor(turns)
        # Reduce around the NEAREST quadrant directly, without first folding to
        # [0, 1). That fold looks harmless and is not: a small negative turn
        # count such as -2.33e-04 becomes 0.99976, which has already discarded
        # the low bits the later subtraction needs, and no amount of core
        # accuracy recovers them. Measured, the fold cost 1283 ulp on small
        # arguments. Reducing straight from `turns` keeps `turns - index/4`
        # exact, because index/4 is a dyadic rational of similar magnitude.
        index = (turns * 4.0 + 0.5).floor()
        residual = (turns - index * 0.25) * TAU
        quadrant = index - (index * 0.25).floor() * 4.0
        return residual, quadrant

    def sin_turns(self, turns: Any) -> Any:
        """``sin(2*pi*turns)``, exact reduction, octant-selected core.

        Each core serves the octant where the OTHER one has its zero. Sine's
        only zero on ``|y| <= pi/4`` is at the origin, which the odd form
        ``y*P(y*y)`` reproduces exactly; cosine never drops below sqrt(2)/2
        there. That is what makes relative accuracy attainable at all -- a
        core whose interval contains its own zero cannot hold a relative
        target near it, at any degree.
        """

        residual, quadrant = self._octant(turns)
        sine = evaluate_core(residual, self.cores["sin"])
        cosine = evaluate_core(residual, self.cores["cos"])
        upper = _where(quadrant == 2.0, -sine, -cosine)
        lower = _where(quadrant == 0.0, sine, cosine)
        return _where(quadrant < 2.0, lower, upper)

    def cos_turns(self, turns: Any) -> Any:
        """``cos(2*pi*turns)`` as a quarter-turn phase; 0.25 shifts exactly."""

        return self.sin_turns(_tensor(turns) + 0.25)

    def tan_turns(self, turns: Any) -> Any:
        return self.sin_turns(turns) / self.cos_turns(turns)

    def sin(self, value: Any) -> Any:
        """Radian entry. Loses a digit per decade of magnitude -- see the
        module docstring's table -- so hold cycles and use ``sin_turns``."""

        return self.sin_turns(_tensor(value) * (1.0 / TAU))

    def cos(self, value: Any) -> Any:
        return self.cos_turns(_tensor(value) * (1.0 / TAU))

    def tan(self, value: Any) -> Any:
        return self.sin(value) / self.cos(value)

    def sec(self, value: Any) -> Any:
        return 1.0 / self.cos(value)

    def csc(self, value: Any) -> Any:
        return 1.0 / self.sin(value)

    def cot(self, value: Any) -> Any:
        return self.cos(value) / self.sin(value)

    def sinc(self, value: Any) -> Any:
        """``sin(x)/x`` on its own even core near zero, where the quotient
        cancels; the direct quotient elsewhere."""

        value = _tensor(value)
        # Band follows the core: only where the quotient cancels.
        inner = _magnitude(value) <= 1.0
        safe_inner = _where(inner, value, value * 0.0)
        near = evaluate_core(safe_inner, self.cores["sinc"])
        safe_outer = _where(inner, value * 0.0 + 1.0, value)
        far = self.sin(safe_outer) / safe_outer
        return _where(inner, near, far)

    # -- inverse circular -------------------------------------------------

    def asin(self, value: Any) -> Any:
        """Own odd core below 1/2; the half-angle identity above it.

        ``asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))`` moves an argument near the
        unbounded-derivative endpoint into the well-conditioned middle, which
        is why the core interval stops at 1/2 rather than reaching 1.
        """

        value = _tensor(value)
        sign = _sign(value)
        magnitude = _magnitude(value)
        inner = magnitude <= 0.5
        safe_inner = _where(magnitude <= 0.5, magnitude, magnitude * 0.0)
        near = evaluate_core(safe_inner, self.cores["asin"])
        reduced = self.sqrt(
            (1.0 - _where(inner, magnitude * 0.0, magnitude)) * 0.5
        )
        far = (reduced * 0.0 + HALF_PI) - 2.0 * evaluate_core(
            reduced, self.cores["asin"],
        )
        return sign * _where(inner, near, far)

    def acos(self, value: Any) -> Any:
        return (_tensor(value) * 0.0 + HALF_PI) - self.asin(value)

    def atan(self, value: Any) -> Any:
        """Two-way split, both halves chosen by the error map.

        ``atan(x) = pi/2 - atan(1/x)`` brings any magnitude down to [0, 1],
        and ``atan(x) = pi/4 + atan((x-1)/(x+1))`` brings [tan(pi/8), 1] down
        to [-tan(pi/8), 0]. The second split is not decoration: the exact
        series does not converge on [0, 1] at all -- 8.2e12 ulp at order 21 --
        and reaches 0.84 ulp on [0, tan(pi/8)] in EIGHTEEN coefficients. The
        narrower interval is both cheaper and better, which is what the
        per-eighth error map predicted when it showed the residual climbing
        toward the interval's top.
        """

        value = _tensor(value)
        sign = _sign(value)
        magnitude = _magnitude(value)

        # First split: reciprocal, down to [0, 1].
        outer = magnitude > 1.0
        safe = _where(outer, magnitude, magnitude * 0.0 + 1.0)
        folded = _where(outer, 1.0 / safe, magnitude)

        # Second split: down to [-tan(pi/8), tan(pi/8)]. The core is odd, so
        # the negative argument the shift produces needs no further handling.
        upper = folded > math.tan(math.pi / 8.0)
        shifted = (folded - 1.0) / (folded + 1.0)
        reduced = _where(upper, shifted, folded)
        core = evaluate_core(reduced, self.cores["atan"])
        inner = _where(upper, core + QUARTER_PI, core)

        extended = _where(outer, (inner * 0.0 + HALF_PI) - inner, inner)
        return sign * extended

    def atan2(self, imaginary: Any, real: Any) -> Any:
        """Full-plane phase; ``angle`` and every unwrap depend on it.

        ``atan`` alone cannot serve a phase -- it collapses the half-planes.
        The quadrant corrections live here rather than in a caller, because a
        caller that gets them subtly wrong produces a phase that is plausible
        everywhere and wrong in one quadrant.
        """

        imaginary, real = _tensor(imaginary), _tensor(real)
        safe = _where(real == 0.0, real * 0.0 + 1.0, real)
        base = self.atan(imaginary / safe)
        rising = imaginary >= 0.0
        shifted = _where(rising, base + math.pi, base - math.pi)
        quadrant = _where(real < 0.0, shifted, base)
        on_axis = _where(
            rising, real * 0.0 + HALF_PI, real * 0.0 - HALF_PI,
        )
        return _where(real == 0.0, on_axis, quadrant)

    # -- exponential ------------------------------------------------------

    def exp(self, value: Any) -> Any:
        """``exp(x) = 2**k * exp(r)`` with ``r`` inside the baked band."""

        value = _tensor(value)
        whole = (value * (1.0 / LN2) + 0.5).floor()
        remainder = value - whole * LN2
        return evaluate_core(remainder, self.cores["exp"]) * (
            (whole * 0.0 + 2.0) ** whole
        )

    def expm1(self, value: Any) -> Any:
        """Own core inside the band; ``exp(x)-1`` outside, where it is safe."""

        value = _tensor(value)
        # The mirror of log1p: w = exp(u) is accurate, but w-1 cancels near
        # zero. w-1 is exact on [1/2, 2], and log(w) recovers the argument
        # that ACTUALLY produced w, so (w-1)*u/log(w) rescales the cancelled
        # difference by the ratio of the true argument to the realised one.
        grown = self.exp(value)
        delta = grown - 1.0
        exact_one = (delta >= 0.0) * (delta <= 0.0) > 0.0
        realised = self.log(_where(exact_one, grown * 0.0 + 1.0, grown))
        safe = _where(exact_one, realised * 0.0 + 1.0, realised)
        return _where(exact_one, value, delta * (value / safe))

    def log(self, value: Any) -> Any:
        """``log(x) = e*ln2 + log(m)`` with ``m`` in the baked mantissa band."""

        value = _tensor(value)
        exponent = _binade(value)
        mantissa = value * ((exponent * 0.0 + 2.0) ** (-exponent))
        # One correction step. Rounding the exponent can leave the mantissa a
        # hair outside the fitted band, where the polynomial EXTRAPOLATES --
        # measured 580 ulp on composed log before this clamp. Nudging the
        # exponent instead keeps the identity exact: x = m * 2**e either way.
        high, low = math.sqrt(2.0), math.sqrt(0.5)
        above, below = mantissa > high, mantissa < low
        exponent = exponent + _where(
            above, exponent * 0.0 + 1.0,
            _where(below, exponent * 0.0 - 1.0, exponent * 0.0),
        )
        mantissa = _where(
            above, mantissa * 0.5,
            _where(below, mantissa * 2.0, mantissa),
        )
        return evaluate_core(mantissa, self.cores["log"]) + exponent * LN2

    def log1p(self, value: Any) -> Any:
        """Own core near zero; ``log(1+u)`` outside, where it does not cancel."""

        value = _tensor(value)
        # w = fl(1+u) is wrong by the rounding of that sum, and near zero that
        # rounding IS the answer. But w-1 is EXACT whenever w lies in [1/2, 2]
        # (Sterbenz), so u/(w-1) is the precise factor by which the sum was
        # spoiled, and log(w) can be corrected by it. This needs no log1p core
        # at all -- the well-conditioned log core carries it -- and it beats
        # the core it replaces by more than an order of magnitude.
        shifted = value + 1.0
        delta = shifted - 1.0
        # delta = 0 exactly when the sum rounded back to 1, where log1p(u) = u.
        exact_one = (delta >= 0.0) * (delta <= 0.0) > 0.0
        safe = _where(exact_one, delta * 0.0 + 1.0, delta)
        return _where(exact_one, value, self.log(shifted) * (value / safe))

    def log10(self, value: Any) -> Any:
        return self.log(value) * (1.0 / LN10)

    def log2(self, value: Any) -> Any:
        return self.log(value) * (1.0 / LN2)

    def sqrt(self, value: Any) -> Any:
        """``sqrt(x) = 2**k * sqrt(m)`` with ``m`` in the baked band."""

        value = _tensor(value)
        exponent = _even_binade(value)
        mantissa = value * ((exponent * 0.0 + 4.0) ** (-exponent))
        above, below = mantissa > 1.0, mantissa < 0.25
        exponent = exponent + _where(
            above, exponent * 0.0 + 1.0,
            _where(below, exponent * 0.0 - 1.0, exponent * 0.0),
        )
        mantissa = _where(
            above, mantissa * 0.25,
            _where(below, mantissa * 4.0, mantissa),
        )
        # Newton, twice. The iteration is a fixed point of sqrt itself, so it
        # is self-correcting and the core only has to be a SEED: measured on
        # the mantissa band, the core alone is 9.7 ulp and two steps take it
        # to 0.79. A better polynomial cannot compete with a rewrite that
        # squares its own correct digits.
        root = evaluate_core(mantissa, self.cores["sqrt"])
        root = 0.5 * (root + mantissa / root)
        root = 0.5 * (root + mantissa / root)
        return root * ((exponent * 0.0 + 2.0) ** exponent)

    def hypot(self, real: Any, imaginary: Any) -> Any:
        """Modulus without the intermediate overflow of ``sqrt(x*x + y*y)``."""

        real, imaginary = _tensor(real), _tensor(imaginary)
        left, right = _magnitude(real), _magnitude(imaginary)
        wider = left > right
        larger = _where(wider, left, right)
        smaller = _where(wider, right, left)
        safe = _where(larger == 0.0, larger * 0.0 + 1.0, larger)
        ratio = smaller / safe
        return _where(
            larger == 0.0, larger * 0.0,
            larger * self.sqrt(ratio * ratio + 1.0),
        )

    # -- hyperbolic -------------------------------------------------------

    def sinh(self, value: Any) -> Any:
        """Own odd core on [-1, 1]; the exp form outside.

        Inside the band ``(exp(x) - exp(-x))/2`` cancels catastrophically, and
        the identity route would also lose the exact oddness the core has by
        construction.
        """

        value = _tensor(value)
        inner = _magnitude(value) <= 1.0
        safe = _where(inner, value, value * 0.0)
        near = evaluate_core(safe, self.cores["sinh"])
        far = (self.exp(value) - self.exp(-value)) * 0.5
        return _where(inner, near, far)

    def cosh(self, value: Any) -> Any:
        value = _tensor(value)
        inner = _magnitude(value) <= 1.0
        safe = _where(inner, value, value * 0.0)
        near = evaluate_core(safe, self.cores["cosh"])
        far = (self.exp(value) + self.exp(-value)) * 0.5
        return _where(inner, near, far)

    def tanh(self, value: Any) -> Any:
        """Own odd core on [-1, 1]; a saturating form outside."""

        value = _tensor(value)
        # Band follows the CORE's interval, which the error map narrowed from
        # 1.0 to 0.5: 2.3e6 ulp against 0.76, in fewer coefficients.
        inner = _magnitude(value) <= 0.5
        safe = _where(inner, value, value * 0.0)
        near = evaluate_core(safe, self.cores["tanh"])
        outer = _where(inner, value * 0.0 + 1.0, value)
        far = 1.0 - 2.0 / (self.exp(2.0 * _magnitude(outer)) + 1.0)
        return _where(inner, near, _sign(value) * far)

    def sech(self, value: Any) -> Any:
        return 1.0 / self.cosh(value)

    def csch(self, value: Any) -> Any:
        return 1.0 / self.sinh(value)

    def coth(self, value: Any) -> Any:
        return self.cosh(value) / self.sinh(value)

    # -- inverse hyperbolic -----------------------------------------------

    def asinh(self, value: Any) -> Any:
        """Own odd core on [-1, 1]; ``log(x + sqrt(x*x + 1))`` outside."""

        value = _tensor(value)
        # Band follows the narrowed core: 9.1e11 ulp on [-1,1], 0.86 on half.
        inner = _magnitude(value) <= 0.5
        safe = _where(inner, value, value * 0.0)
        near = evaluate_core(safe, self.cores["asinh"])
        outer = _where(inner, value * 0.0 + 1.0, _magnitude(value))
        far = self.log(outer + self.sqrt(outer * outer + 1.0))
        return _where(inner, near, _sign(value) * far)

    def acosh(self, value: Any) -> Any:
        """``log(x + sqrt(x*x - 1))``. No core: the branch point at 1 is not
        removable by a polynomial, and the log form is well conditioned above
        it."""

        value = _tensor(value)
        safe = _where(value < 1.0, value * 0.0 + 1.0, value)
        return self.log(safe + self.sqrt(safe * safe - 1.0))

    def atanh(self, value: Any) -> Any:
        """Own odd core below 1/2; ``0.5*log1p(2u/(1-u))`` outside."""

        value = _tensor(value)
        magnitude = _magnitude(value)
        inner = magnitude <= 0.5
        safe = _where(inner, value, value * 0.0)
        near = evaluate_core(safe, self.cores["atanh"])
        outer = _where(inner, magnitude * 0.0, magnitude)
        far = 0.5 * self.log1p(2.0 * outer / (1.0 - outer))
        return _where(inner, near, _sign(value) * far)

    # -- complex, over component pairs ------------------------------------

    def cis_turns(self, turns: Any) -> tuple[Any, Any]:
        """``exp(i*2*pi*turns)`` as a (real, imaginary) pair -- the twiddle.

        Returned as a component pair rather than a complex tensor because
        complex support is ragged across backends; the algebra below stays on
        the same pairs.
        """

        return self.cos_turns(turns), self.sin_turns(turns)

    def modulus(self, real: Any, imaginary: Any) -> Any:
        return self.hypot(real, imaginary)

    def angle(self, real: Any, imaginary: Any) -> Any:
        return self.atan2(imaginary, real)

    def conjugate(self, real: Any, imaginary: Any) -> tuple[Any, Any]:
        return _tensor(real), -_tensor(imaginary)

    def multiply(self, left: tuple[Any, Any],
                 right: tuple[Any, Any]) -> tuple[Any, Any]:
        left_real, left_imaginary = left
        right_real, right_imaginary = right
        return (
            left_real * right_real - left_imaginary * right_imaginary,
            left_real * right_imaginary + left_imaginary * right_real,
        )


#: Every method the surface offers, in catalogue order.
SURFACE_METHODS = (
    "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "asinh", "acosh", "atanh", "sinc",
)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


_ACTIVE: dict[str, SignalMath] = {}


def signal_math(quality: str = "reference") -> SignalMath:
    """The surface at one named quality, baked once and reused."""

    existing = _ACTIVE.get(str(quality))
    if existing is not None:
        return existing
    surface = SignalMath(prebake(quality))
    _ACTIVE[str(quality)] = surface
    return surface


__all__ = [
    "AnglePalette",
    "bake_angle_palette",
    "fit_exact",
    "SELECTABLE_FAMILIES",
    "fit_best",
    "CORE_RANGES",
    "DEFAULT_FAMILY",
    "DEFAULT_MODE",
    "FAMILIES",
    "GROWTH_LIMITS",
    "MODES",
    "PREBAKE_SETS",
    "PRIMITIVES",
    "SURFACE_METHODS",
    "BakedCore",
    "CoreRange",
    "CoreSet",
    "PrebakeSettings",
    "SignalMath",
    "evaluate_core",
    "fit_core",
    "fit_lut",
    "fit_polyspline",
    "fit_series",
    "fit_structured",
    "prebake",
    "signal_math",
    "validate_epsilon",
]
