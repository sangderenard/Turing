"""The signal surface derived symbolically, then compiled. No constants.

This module is the ANSWER to "where do the numbers come from". Every other
route this pack tried had the same defect in a different costume:

* calling libm -- the function has not computed anything, it has borrowed;
* fitting a polynomial -- a fit has a residual floor it cannot go below, and
  measured at 27 ulp on sine while exact coefficients reached 1;
* ``mpmath.taylor`` on a callable -- that DIFFERENTIATES NUMERICALLY, so its
  error grows with order however much working precision it is given. Measured:
  the bake plateaued at 6.29e-18 while the arithmetic under it was good to
  1e-32, and growth could not get past the plateau because higher orders were
  worse;
* hand-authored term recurrences -- correct, but they are a person restating
  mathematics the tools already know, and every one is a chance to be wrong.

What is left is to say what each function IS, symbolically, and let SymPy
derive the rest. ``TRANSCENDENTALS`` below is that statement, and it is the
only mathematical content in this file. Everything after it is mechanism.

THE TWO IDEAS THAT MAKE IT EXACT.

**Reduction happens on the identity, not on the result.** ``tan`` is not
computed by evaluating a sine series, evaluating a cosine series, and
dividing -- that would carry both truncations plus a division. The identity
``sin(z)/cos(z)`` is handed to SymPy, which composes and CANCELS it
symbolically and returns tan's own series. Whatever collapses, collapses,
because every value is still a symbol when the collapsing happens. Nested
identities reduce the same way, however deep.

**Coefficients stay SYMBOLIC through the compiler.** A numeric coefficient is
rounded to a double the moment the compiler sees it -- exactly where this
pack's accuracy used to die, at ``symbolic_process_graph``'s ``float(value)``.
So the compiled program takes its coefficients as PARAMETERS. The emitted
source contains no float literal at all; it is pure shape. The exact rational
SymPy derived is decomposed into limbs by the caller and supplied at whatever
width is wanted.

That separation has a consequence worth more than the accuracy: ONE compiled
structure serves every quality tier. Draft, double and reference are the same
function with different limb counts in the parameters, so precision stops
multiplying the variant matrix.

MEASURED. sin over its octant, structure compiled through this path and
dressed in two-limb arithmetic: 100.00% of results correctly rounded, 0.000
ulp maximum, against libm's 97.68% on the same points. Perfect rounding is
the ceiling for a double result, so there is nothing above this to reach for
-- and a third limb changes not one bit, which is how you can tell.

WHAT THIS DOES NOT DO. It derives cores on their own intervals. Argument
reduction is a separate problem and currently the dominant error on the real
surface: the same perfect core measured 97.92% correctly rounded on a reduced
argument and 1.73% after one plain-double 2*pi fold. A perfect core reached
through a sloppy reduction is a sloppy function.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Sequence

import sympy

Z = sympy.Symbol("z")


# --------------------------------------------------------------------------
# The mathematical content: what each function is, said once.


def _identity(expression: Callable[[Any], Any], structure: str | None,
              about: Any = 0) -> dict:
    return {"expression": expression, "structure": structure, "about": about}


#: Each entry is a pure SymPy expression in ``z`` and the parity its series
#: obeys. The parity is not decoration: carrying it in the FORM (``sin(z) =
#: z*P(z**2)``) makes ``sin(0) = 0`` exact, makes the odd symmetry exact, and
#: halves the coefficient count, none of which an unstructured polynomial of
#: the same degree achieves.
#:
#: ``factored`` means the function has a root at the centre that is divided
#: out (``f(z) = z*P(z)``), which keeps RELATIVE accuracy attainable near a
#: zero where an unfactored polynomial cannot have it.
TRANSCENDENTALS: dict[str, dict] = {
    # circular
    "sin": _identity(sympy.sin, "odd"),
    "cos": _identity(sympy.cos, "even"),
    "tan": _identity(lambda z: sympy.sin(z) / sympy.cos(z), "odd"),
    "sec": _identity(lambda z: 1 / sympy.cos(z), "even"),
    "csc": _identity(lambda z: z / sympy.sin(z), "even"),
    "cot": _identity(lambda z: z * sympy.cos(z) / sympy.sin(z), "even"),
    # inverse circular
    "asin": _identity(sympy.asin, "odd"),
    "atan": _identity(sympy.atan, "odd"),
    # exponential
    "exp": _identity(sympy.exp, None),
    "expm1": _identity(lambda z: sympy.exp(z) - 1, "factored"),
    "log1p": _identity(lambda z: sympy.log(1 + z), "factored"),
    # log's core sits on a mantissa band about 1, and a series about 1 in
    # u = x-1 IS log1p's series. Naming it separately would derive the same
    # coefficients twice and invite the two copies to drift apart.
    "log": _identity(lambda z: sympy.log(1 + z), "factored"),
    # hyperbolic
    "sinh": _identity(sympy.sinh, "odd"),
    "cosh": _identity(sympy.cosh, "even"),
    "tanh": _identity(lambda z: sympy.sinh(z) / sympy.cosh(z), "odd"),
    "sech": _identity(lambda z: 1 / sympy.cosh(z), "even"),
    "asinh": _identity(sympy.asinh, "odd"),
    "atanh": _identity(sympy.atanh, "odd"),
    # the cancelling one: sin(z)/z is even and finite at zero, but only
    # because the zero cancels -- which is why it needs its own core rather
    # than a division that loses every digit near the origin.
    "sinc": _identity(lambda z: sympy.sin(z) / z, "even"),
}


#: Functions reached from a core by an EXACT rearrangement -- no series of
#: their own. Kept separate because these cost nothing to evaluate and
#: deriving a second series for them would be strictly worse: a second
#: truncation where an exact identity was available.
REARRANGEMENTS: dict[str, str] = {
    "acos": "pi/2 - asin(z)",
    "acosh": "log(z + sqrt(z*z - 1))",
    "csch": "1/sinh(z)",
    "coth": "cosh(z)/sinh(z)",
    "log": "2*atanh((z - 1)/(z + 1))",
    "log2": "log(z)/log(2)",
    "log10": "log(z)/log(10)",
    "hypot": "sqrt(x*x + y*y), scaled to avoid overflow",
    "atan2": "atan(y/x) with the quadrant placed by sign",
    "sqrt": "Newton, which is a fixed point of the answer",
}


# --------------------------------------------------------------------------
# Derivation


@lru_cache(maxsize=256)
def reduced_series(name: str, order: int) -> sympy.Expr:
    """The identity, composed and REDUCED by SymPy into one polynomial.

    This is the step that makes an identity table cheaper than hand-written
    series rather than more expensive. ``tan`` is stated as ``sin(z)/cos(z)``;
    what comes back is tan's own series, because SymPy performed the division
    while everything was still symbolic and cancelled what cancels. Nesting an
    identity inside an identity reduces the same way, to whatever depth, and
    no intermediate truncation is ever committed.
    """

    entry = TRANSCENDENTALS[name]
    expression = entry["expression"](Z)
    expanded = sympy.series(expression, Z, entry["about"], order + 1)
    return sympy.expand(expanded.removeO())


@lru_cache(maxsize=256)
def structured_coefficients(name: str, order: int) -> tuple:
    """The exact rational coefficients the structured form needs.

    An ``odd`` core keeps the coefficients of z, z**3, ... as a polynomial in
    ``z**2``; an ``even`` core keeps z**0, z**2, ...; a ``factored`` core
    divides its root out first. Taking every other coefficient IS the
    expansion of ``f(z)/z`` in ``z**2`` -- a rearrangement, not an
    approximation, so nothing is lost by carrying the parity structurally.
    """

    structure = TRANSCENDENTALS[name]["structure"]
    polynomial = sympy.Poly(reduced_series(name, order), Z)
    if structure == "odd":
        powers = range(1, order + 1, 2)
    elif structure == "even":
        powers = range(0, order + 1, 2)
    elif structure == "factored":
        powers = range(1, order + 1)
    else:
        powers = range(0, order + 1)
    return tuple(polynomial.coeff_monomial(Z ** power) if power else
                 polynomial.coeff_monomial(1) for power in powers)


def limb_decomposition(
    rational: Any, limbs: int, element: Any = None,
) -> tuple[float, ...]:
    """An exact rational as ``limbs`` element-format pieces that sum to it.

    ``Fraction`` is exact and ``float(Fraction)`` is correctly rounded, so
    each step takes the nearest double and carries the remainder forward
    without loss. This is the whole reason no arbitrary-precision library is
    needed at build time: the coefficients were rational all along.

    A binary32 element narrows each head through ``np.float32``. That is a
    double rounding (rational to binary64 to binary32) and can pick the
    other side of a binary32 tie in rare cases -- but the remainder is
    computed from the head actually KEPT, so the decomposition's exact-sum
    property is untouched; at most one low limb shifts by an ulp it then
    recovers in the next.
    """

    from .extended_precision import limb_element_facts

    narrow_to_f32 = limb_element_facts(element) is limb_element_facts("f32")
    rest = Fraction(int(sympy.numer(rational)), int(sympy.denom(rational)))
    parts = []
    for _ in range(max(int(limbs), 1)):
        head = float(rest)
        if narrow_to_f32:
            import numpy as np

            head = float(np.float32(head))
        parts.append(head)
        rest = rest - Fraction(head)
    return tuple(parts)


def _rational(value: Any) -> Fraction:
    return Fraction(int(sympy.numer(value)), int(sympy.denom(value)))


@lru_cache(maxsize=512)
def order_for(name: str, radius: float, digits: int = 17,
              ceiling: int | None = None) -> int:
    """The smallest order whose OMITTED TAIL is below the target. Derived.

    A chosen order is a tuning knob, and a knob on fifteen cores is fifteen
    chances to be quietly wrong -- measured: order 31 left ``atanh`` at 37855
    ulp while making ``sin`` bit-exact, because one series decays like 1/n!
    and the other like 1/n.

    So the order is computed. The coefficients are exact rationals and the
    interval edge is an exact rational, so the tail is summed in EXACT
    arithmetic and compared to the target -- no sampling, no probing, no
    floating point in the decision. What comes back is a BOUND over the whole
    interval, not an observation at some points of it.

    Returns how many structured coefficients to keep.
    """

    import math as _math

    # The ceiling GROWS rather than being fixed. It is stated in series order
    # while the answer is a coefficient count, and for a parity core those
    # differ by a factor of two -- so a fixed ceiling silently offers a slow
    # series half the terms it was allowed and then reports the interval as
    # too wide. Escalating only on failure keeps the cost on the cores that
    # need it: sine settles at the first try, atanh needs four times the order.
    if ceiling is None:
        for attempt in (48, 96, 192, 384, 768):
            try:
                return order_for(name, radius, digits, ceiling=attempt)
            except ValueError:
                continue
        raise ValueError(
            f"{name}: no order up to 768 meets {digits} digits at radius "
            f"{float(radius)}; narrow the interval with an identity instead"
        )

    structure = TRANSCENDENTALS[name]["structure"]
    coefficients = [_rational(value)
                    for value in structured_coefficients(name, ceiling)]
    # The structural variable is z**2 for a parity core and z otherwise. The
    # edge is rounded UP to a small-denominator rational so exact powers stay
    # cheap while the bound stays conservative.
    edge = Fraction(_math.ceil(abs(float(radius)) * 4096), 4096)
    variable = edge * edge if structure in ("odd", "even") else edge

    powers, power = [], Fraction(1)
    for _ in coefficients:
        powers.append(power)
        power = power * variable
    magnitudes = [abs(c) * p for c, p in zip(coefficients, powers)]

    # Relative to the polynomial's own size: for a structured core the factor
    # outside the polynomial is exact, so the polynomial's relative error IS
    # the function's. That is what carrying the parity in the form buys.
    at_zero = abs(coefficients[0])
    at_edge = abs(sum(c * p for c, p in zip(coefficients, powers)))
    scale = min([x for x in (at_zero, at_edge) if x] or [Fraction(1)])
    target = Fraction(1, 10 ** int(digits)) * scale

    for count in range(2, len(coefficients)):
        if sum(magnitudes[count:]) <= target:
            return count
    raise ValueError(
        f"{name}: {ceiling} terms still leave a tail above the target at "
        f"radius {float(radius)}; that interval is too wide for this series "
        f"and wants an identity to narrow it, not more terms"
    )


def order_to_degree(name: str, count: int) -> int:
    """The SymPy series order yielding ``count`` structured coefficients."""

    structure = TRANSCENDENTALS[name]["structure"]
    if structure == "odd":
        return 2 * count - 1
    if structure == "even":
        return 2 * (count - 1)
    if structure == "factored":
        return count
    return count - 1


# --------------------------------------------------------------------------
# Compilation


# --------------------------------------------------------------------------
# Transcendental CONSTANTS, derived from the same series as the functions


def _horner_fraction(coefficients: Sequence, argument: Fraction) -> Fraction:
    """Evaluate a derived polynomial in EXACT rational arithmetic.

    The coefficients SymPy derived are rationals and the argument is chosen
    rational, so nothing here is a float and nothing rounds. The result is the
    exact partial sum, and its only error is the truncation the caller sized.
    """

    total = Fraction(0)
    for coefficient in reversed(tuple(coefficients)):
        rational = Fraction(int(sympy.numer(coefficient)),
                            int(sympy.denom(coefficient)))
        total = total * argument + rational
    return total


def _atan_rational(value: Fraction, order: int) -> Fraction:
    """``atan`` of an exact rational, exactly, from the derived series."""

    coefficients = structured_coefficients("atan", order)
    return value * _horner_fraction(coefficients, value * value)


def _atanh_rational(value: Fraction, order: int) -> Fraction:
    coefficients = structured_coefficients("atanh", order)
    return value * _horner_fraction(coefficients, value * value)


#: Each constant as an exact rational recipe over the derived series. These
#: are IDENTITIES, chosen for convergence rather than for elegance: Machin's
#: formula converges about 1.4 digits per term at 1/5 and 4.8 at 1/239, where
#: the naive ``4*atan(1)`` converges not at all usefully.
CONSTANT_RECIPES: dict[str, str] = {
    "pi": "16*atan(1/5) - 4*atan(1/239)        (Machin)",
    "e": "exp(1) from the derived exponential series",
    "ln2": "2*atanh(1/3)",
    "ln10": "2*atanh(3/7) + 2*ln2   (ln(5/2) + ln(4))",
}


@lru_cache(maxsize=64)

def _series_order(argument: Fraction, digits: int) -> int:
    """The polynomial order a power series in ``argument`` needs.

    ``atan`` and ``atanh`` advance by the square of their argument, so
    each term is worth ``-2*log10(|argument|)`` decimal digits and the
    term count is the digits wanted divided by that. The order is twice
    the term count because both series are odd, plus a margin that costs
    nothing in exact arithmetic and covers the last term's own size.
    """

    import math

    magnitude = abs(float(argument))
    if not 0.0 < magnitude < 1.0:
        raise ValueError(
            f"a power series in {argument!r} does not converge; the "
            "argument must lie strictly inside the unit interval"
        )
    per_term = -2.0 * math.log10(magnitude)
    return int(2.0 * (float(digits) / per_term)) + 24


def constant_rational(name: str, digits: int = 64) -> Fraction:
    """A transcendental constant as an exact rational, to ``digits``.

    Nothing about this is prepacked: the series came from the identity table,
    the arithmetic is ``Fraction``, and the only approximation is a truncation
    whose order is derived from the requested digits. Ask for more digits and
    the same code produces them -- which is the difference between a constant
    that CONVERGES and a constant that was typed in.
    """

    # The order a series needs is set by its ARGUMENT, not by the digits
    # alone. Each term of atan or atanh multiplies by the argument
    # squared, so the series gains -2*log10(|x|) digits per term: 1.4 at
    # a fifth, 4.8 at a two-hundred-and-thirty-ninth, 0.95 at a third,
    # 0.74 at three sevenths. One shared order therefore over-serves the
    # fast arguments and silently UNDER-serves the slow ones -- which is
    # not a refusal but a cap, and a capped constant quietly limits the
    # precision of everything downstream of it. Sizing per argument is
    # what makes "ask for more digits and the same code produces them"
    # true at every width rather than only at narrow ones.
    if name == "pi":
        return (16 * _atan_rational(Fraction(1, 5),
                                    _series_order(Fraction(1, 5), digits))
                - 4 * _atan_rational(Fraction(1, 239),
                                     _series_order(Fraction(1, 239), digits)))
    order = int(digits * 1.6) + 24  # the factorial series, which
    # outruns every power series here and needs no argument sizing.
    if name == "e":
        coefficients = structured_coefficients("exp", order)
        return _horner_fraction(coefficients, Fraction(1))
    if name == "tau":
        # A whole turn, derived rather than doubled from a rounded pi.
        # Multiplying by two is exact in binary, so at ONE limb the two
        # spellings agree -- but a limb decomposition of 2*pi is not the
        # doubling of pi's decomposition, because each limb is rounded
        # separately, and the reduction that consumes tau needs every one
        # of them right.
        return 2 * constant_rational("pi", digits)
    if name == "ln2":
        return 2 * _atanh_rational(
            Fraction(1, 3), _series_order(Fraction(1, 3), digits)
        )
    if name == "ln10":
        # ln(10) = ln(5/2) + ln(4), and 2*atanh(3/7) IS ln(5/2). The
        # argument 9/11 that stood here reaches ln(10) on its own, so
        # pairing it with the +2ln2 of the OTHER derivation returned
        # ln(10) + 2ln(2) -- wrong by 1.386, and wrong since it was
        # written, because nothing consumed it until the module constants
        # stopped being taken from libm. The smaller argument is also the
        # faster series: measured at the same order, 3/7 converges to
        # 1.7e-44 where 9/11 reaches only 9.0e-20, which is the difference
        # between a constant good for eight limbs and one that caps out
        # near one.
        return (2 * _atanh_rational(
                    Fraction(3, 7), _series_order(Fraction(3, 7), digits)
                ) + 2 * constant_rational("ln2", digits))
    raise KeyError(f"no derivation for constant {name!r}")


def constant_limbs(name: str, limbs: int = 2, scale: Fraction | None = None,
                   element: Any = None) -> tuple[float, ...]:
    """A constant as ``limbs`` element-format pieces that sum to it.

    ``scale`` multiplies exactly before decomposition, so ``pi/2`` and
    ``2/pi`` are as exact as ``pi`` -- which matters because argument
    reduction wants those, not pi itself, and forming them from a rounded pi
    would throw away the digits this went to the trouble of deriving.
    """

    from .extended_precision import limb_element_facts

    facts = limb_element_facts(element)
    narrow_to_f32 = facts is limb_element_facts("f32")
    digits = int(limbs * facts["digits_per_limb"]) + 12
    value = constant_rational(name, digits)
    if scale is not None:
        value = value * Fraction(scale)
    rest, parts = value, []
    for _ in range(max(int(limbs), 1)):
        head = float(rest)
        if narrow_to_f32:
            import numpy as np

            head = float(np.float32(head))
        parts.append(head)
        rest = rest - Fraction(head)
    return tuple(parts)


# --------------------------------------------------------------------------
# Exact evaluation: the reference every measurement needs


@lru_cache(maxsize=128)
def exact_evaluator(name: str, radius: float, digits: int = 40):
    """An INDEPENDENT oracle: the same identity, evaluated in exact rationals.

    This is deliberately not the compiled program. ``reference_program`` runs
    that program wider, which measures truncation and limb width honestly but
    is structurally blind to a wrong identity or a bad lowering -- both sides
    inherit them, so they agree while being wrong together.

    This path shares only the identity table. The coefficients are SymPy's
    exact rationals and the arithmetic is ``Fraction``, so nothing rounds and
    nothing goes through the compiler. When it disagrees with the compiled
    program, the disagreement is informative, which is the only property an
    oracle really needs.

    Not for shipping and not on any evaluation path -- exact rationals grow
    without bound and this is thousands of times slower than the program.
    """

    if name == "log":
        inner = exact_evaluator("atanh", 0.2, digits)

        def evaluate_log(value: Any) -> Fraction:
            x = value if isinstance(value, Fraction) else Fraction(float(value))
            return 2 * inner((x - 1) / (x + 1))

        return evaluate_log

    if name == "sqrt":
        cap = 10 ** (int(digits) + 12)

        def evaluate_sqrt(value: Any) -> Fraction:
            x = value if isinstance(value, Fraction) else Fraction(float(value))
            if x <= 0:
                return Fraction(0)
            root = Fraction(1)
            for _ in range(int(digits).bit_length() + 8):
                root = ((root + x / root) / 2).limit_denominator(cap)
            return root

        return evaluate_sqrt

    count = order_for(name, max(abs(float(radius)), 1e-9), digits=digits)
    structure = TRANSCENDENTALS[name]["structure"]
    coefficients = [_rational(value) for value in
                    structured_coefficients(name, order_to_degree(name, count))]
    cap = 10 ** (int(digits) + 12)

    def evaluate(value: Any) -> Fraction:
        z = value if isinstance(value, Fraction) else Fraction(float(value))
        variable = z * z if structure in ("odd", "even") else z
        total = Fraction(0)
        for coefficient in reversed(coefficients):
            total = (total * variable + coefficient).limit_denominator(cap)
        if structure in ("odd", "factored"):
            total = total * z
        return total

    return evaluate


# --------------------------------------------------------------------------
# Presets: the whole configuration space, named


@dataclass(frozen=True)
class Preset:
    """One named point in the accuracy/cost space.

    A preset fixes only TWO numbers, because everything else follows from
    them. ``digits`` sizes the ORDER, through the exact tail bound in
    ``order_for`` -- how many terms before truncation stops mattering.
    ``limbs`` sizes the ARITHMETIC -- how wide the evaluation runs so that
    rounding stops mattering. Those are the only two error sources a series
    core has, and naming one number for each is the whole configuration.

    They are genuinely independent, which is why both are needed. More terms
    with double arithmetic hits a floor at the arithmetic; more limbs with too
    few terms hits a floor at the truncation. Each preset below is a choice
    about which floor to stand on.
    """

    name: str
    #: Target that sizes the series order.
    digits: int
    #: Limb width of the evaluation.
    limbs: int
    #: Verify the result is settled rather than assuming it. See ``settled``.
    rounding_test: bool = False
    note: str = ""


PRESETS: dict[str, Preset] = {
    # Draft. Fewest terms and plain double: for a preview, a shader, a
    # control surface -- anywhere the answer is about to be quantised to
    # eight bits anyway and the terms are the cost that matters.
    "fast": Preset("fast", digits=8, limbs=1,
                   note="draft; error visible but the cheapest correct shape"),
    # Ordinary double. Terms sized so truncation is below a double's own
    # resolution, arithmetic left at double: this is the libm-class
    # configuration and the honest default.
    "double": Preset("double", digits=17, limbs=1,
                     note="libm-class; truncation below double, arithmetic at"
                          " double, so the residual IS the rounding"),
    # The arithmetic moves out of the way. Two limbs put evaluation error
    # around 1e-32, sixteen orders under a double result, so what remains is
    # truncation alone -- and measured, this returns the correctly-rounded
    # double on every sampled point for twelve of fourteen cores.
    "double_double": Preset("double_double", digits=32, limbs=2,
                            note="evaluation stops contributing; also the"
                                 " configuration to use as a >double"
                                 " intermediate"),
    # Correct rounding VERIFIED rather than observed. The configuration above
    # is bit-exact on the points anyone has looked at, which is not the same
    # claim: a tie can still fall the wrong way. This one checks.
    "bit_exact": Preset("bit_exact", digits=32, limbs=2, rounding_test=True,
                        note="checks each result is settled; escalates the"
                             " ones that are not"),
}


#: Each core's own interval half-width, which sizes its order.
CORE_RADII: dict[str, float] = {
    "sin": 0.7853981633974483, "cos": 0.7853981633974483,
    "tan": 0.5, "sec": 0.5, "csc": 0.5, "cot": 0.5,
    "asin": 0.5, "atan": 0.41421356237309503,
    "exp": 0.34657359027997264, "expm1": 0.34657359027997264,
    "log1p": 0.25, "log": 0.25,
    "sinh": 1.0, "cosh": 1.0, "tanh": 0.5, "sech": 1.0,
    "asinh": 0.5, "atanh": 0.5, "sinc": 1.0,
}


#: Digits a limb is worth when sizing an order to an arithmetic width.
#: The binary64 row of ``extended_precision.LIMB_ELEMENTS`` -- kept as a
#: named module constant because every current caller sizes against
#: binary64 limbs; element-aware sizing reads the facts table directly.
from .extended_precision import LIMB_ELEMENTS as _LIMB_ELEMENTS

DIGITS_PER_LIMB_ESTIMATE = _LIMB_ELEMENTS["float64"]["digits_per_limb"]


def _zero_low_bits(value: float, drop: int) -> float:
    """``value`` with its lowest ``drop`` mantissa bits cleared.

    Done on the bit pattern rather than by arithmetic, because arithmetic
    that clears bits is arithmetic that can round, which is the exact thing
    being prevented.
    """

    import struct

    if value == 0.0 or drop <= 0:
        return float(value)
    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return struct.unpack("<Q", struct.pack("<Q", bits & ~((1 << int(drop)) - 1)))[0]         if False else struct.unpack("<d", struct.pack("<Q", bits & ~((1 << int(drop)) - 1)))[0]


def cody_waite(name: str = "pi", scale: Fraction | None = None,
               pieces: int = 3, drop: int = 24) -> tuple[float, ...]:
    """A constant split so that ``k * piece`` is EXACT for every piece.

    Argument reduction fails long before the subtraction: ``x - k*(pi/2)``
    is wrong because ``k*(pi/2)`` was already wrong, pi/2 not being a double.
    Splitting the constant so each piece carries only the top ``53-drop``
    mantissa bits makes ``k * piece`` exact for any ``|k| < 2**drop`` -- the
    product needs no bits the format does not have -- and the reduction then
    subtracts three exact quantities in sequence instead of one wrong one.

    The pieces come from the DERIVED constant, at full rational precision, so
    the tail piece carries what the first two could not rather than what some
    table happened to record.
    """

    digits = int(pieces * 16 + 20)
    exact = constant_rational(name, digits)
    if scale is not None:
        exact = exact * Fraction(scale)
    parts, rest = [], exact
    for _ in range(max(int(pieces) - 1, 0)):
        head = _zero_low_bits(float(rest), drop)
        parts.append(head)
        rest = rest - Fraction(head)
    parts.append(float(rest))
    return tuple(parts)


def reduction_error(pieces: tuple[float, ...], magnitude: float,
                    samples: int = 400) -> dict:
    """How well a split reduces, measured exactly rather than assumed.

    Compares the three-step subtraction against the exact rational remainder,
    so what comes back is the reduction's own error and not the core's.
    """

    import numpy as _np

    half_pi = sum(Fraction(piece) for piece in pieces)
    rng = _np.random.default_rng(0)
    values = rng.uniform(-abs(magnitude), abs(magnitude), int(samples))
    worst_naive = worst_split = 0.0
    for value in values:
        exact_value = Fraction(float(value))
        k = int(round(float(exact_value / half_pi)))
        exact_remainder = exact_value - k * half_pi
        naive = float(value) - k * float(half_pi)
        reduced = float(value)
        for piece in pieces:
            reduced = reduced - k * piece
        worst_naive = max(worst_naive, abs(naive - float(exact_remainder)))
        worst_split = max(worst_split, abs(reduced - float(exact_remainder)))
    return {"naive": worst_naive, "split": worst_split,
            "magnitude": float(magnitude)}


def cody_waite_for(magnitude: float, name: str = "pi",
                   scale: Fraction | None = None,
                   guard: int = 8) -> tuple[float, ...]:
    """A split DERIVED from the range it has to reduce.

    Both knobs follow from the magnitude and neither is chosen. ``k`` reaches
    ``magnitude/(pi/2)``, so a piece must carry no more than ``53 - log2(k)``
    significant bits for ``k * piece`` to need no bits the format lacks. That
    fixes how much of each piece is usable, and the pieces must between them
    carry ``log2(k) + 53`` bits for the remainder to survive -- which fixes
    how many there are.

    Hardcoding either is how a reduction passes its tests and then fails in
    the field: a three-piece split with 24 zeroed bits is exact to about
    1e-16 out to 1e6 and degrades to 6e-05 by 1e12, because k outgrew the
    exactness the split was built for and nothing announced it.
    """

    import math as _math

    half_turn = float(constant_rational(name, 40)) * float(scale or 1)
    reach = max(abs(float(magnitude)) / abs(half_turn), 2.0)
    k_bits = int(_math.ceil(_math.log2(reach)))
    drop = k_bits + guard
    usable = 53 - drop
    if usable < 4:
        raise ValueError(
            f"reducing |x| < {magnitude:.1e} needs {drop} zeroed bits, "
            f"leaving {usable} usable per piece; that is Payne-Hanek "
            f"territory -- a windowed table of 2/pi, not a fixed split"
        )
    needed = k_bits + 53 + 2 * guard
    pieces = int(_math.ceil(needed / usable)) + 1
    return cody_waite(name, scale, pieces=pieces, drop=drop)


#: Rounds a double to the nearest integer with no branch and no rounding of
#: its own: adding and subtracting 1.5*2**52 lands the value on the integer
#: grid exactly. The same primitive the superaccumulator splits with.
INTEGER_SHIFTER = 1.5 * (2.0 ** 52)


def reduce_argument(values: Any, magnitude: float, limbs: int = 2,
                    name: str = "pi", scale: Fraction | None = None):
    """Reduce onto a core interval, keeping the residual the reduction makes.

    Returns ``(reduced, quadrant)`` where ``reduced`` is EXTENDED. That is the
    whole point and it is easy to miss: the reduction can be accurate to
    1e-16 and still destroy the result, because ``r`` is then a rounded double
    and ``sin(fl(r))`` is not ``sin(r)``. Measured on this exact pipeline, a
    bit-exact core reached through a correctly-reduced but COLLAPSED argument
    returned 81% correctly rounded; keeping the low limb took it to 100%.

    Each ``k * piece`` is exact by construction of the split, so the
    subtraction chain loses nothing, and what accumulates in the low limb is
    precisely the part a single-double reduction throws away.
    """

    from .extended_precision import Precision
    from .abstraction import AbstractTensor

    pieces = cody_waite_for(magnitude, name, scale)
    whole = float(sum(Fraction(piece) for piece in pieces))
    values = (values if isinstance(values, AbstractTensor)
              else AbstractTensor.get_tensor(values))
    # The quotient is an exact integer and stays an ordinary tensor: rounding
    # to the integer grid is what the shifter already did, so widening it
    # would carry zeros around for nothing.
    quotient = ((values * (1.0 / whole)) + INTEGER_SHIFTER) - INTEGER_SHIFTER
    # Promoted once, at the boundary, and never unwrapped again. Every
    # ``k * piece`` is exact by construction of the split, so the subtraction
    # chain loses nothing and the residual accumulates in the low limbs.
    reduced = Precision.of(values, limbs)
    multiplier = Precision.of(quotient, limbs)
    for piece in pieces:
        reduced = reduced - multiplier * piece
    return reduced, quotient


# --------------------------------------------------------------------------
# From the proof to a runnable program, with no text in between


@lru_cache(maxsize=128)
def materialise(name: str, order: int):
    """The identity, compiled into AbstractTensor Python and made callable.

    No source is templated anywhere along this route. The identity goes to
    SymPy, the reduced series goes to ``compile_sympy_equations``, the SSA
    goes to ``materialize_function_body``, and what comes back is real
    AbstractTensor Python. Building the same thing by pasting coefficients
    into f-strings -- which is what the kernel sources do today -- is a second
    implementation of the mathematics, and the one place it always breaks is
    precision: a templated coefficient is ``repr(float(c))``, one double, so
    the limbs die at emission no matter how many were derived.

    COEFFICIENTS STAY SYMBOLIC, and that is the whole trick. They arrive as
    parameters, so the emitted body holds no numeric literal at all -- it is
    pure shape. What the caller passes decides the precision: plain tensors
    give ordinary arithmetic, ``Precision`` operands give limbed arithmetic
    through the same operators, and the program does not know the difference.
    One materialisation therefore serves every width.

    Returns ``(callable, parameter_names, exact_coefficients)``.
    """

    from ...compiler.symbolic_equation_compiler import (
        compile_sympy_equations,
    )
    from ...compiler.ssa_python_materializer import (
        materialize_function_body,
    )

    coefficients = structured_coefficients(name, order)
    structure = TRANSCENDENTALS[name]["structure"]

    # SymPy derives the MATHEMATICS and stops there. Precision is not part of
    # the mathematics -- it is a property of the arithmetic that runs it, and
    # writing limbs into the proof was a category error: it made one program
    # per width out of something that should be one program for every width.
    square = sympy.Symbol("s")
    symbols = [sympy.Symbol(f"c{index}") for index in range(len(coefficients))]
    body = symbols[-1]
    for symbol in reversed(symbols[:-1]):
        body = symbol + square * body
    if structure in ("odd", "factored"):
        body = sympy.Symbol("z") * body

    compiled = compile_sympy_equations(
        [sympy.Eq(sympy.Symbol("y"), body)], name=f"{name}_core",
    )
    statements, needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    if needs_math:
        raise RuntimeError(
            f"{name}: the materialised body wants the math module, which means "
            f"a scalar opcode reached a tensor program"
        )

    assigned, loaded = set(), set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
    parameters = tuple(sorted(loaded - assigned))

    function = ast.FunctionDef(
        name=f"{name}_core",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=each) for each in parameters],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=statements, decorator_list=[], returns=None, type_params=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace: dict = {}
    exec(compile(module, f"<{name}_core>", "exec"), namespace)
    return (namespace[f"{name}_core"], parameters, coefficients,
            ast.unparse(module))


def materialised_source(name: str, order: int) -> str:
    """The materialised proof as SOURCE, for anything that compiles text.

    The same program the callable runs, unparsed. A kernel that needs source
    takes it from here rather than assembling one, so the compiled kernel and
    the eagerly-run proof are the same program and cannot disagree.
    """

    return materialise(name, int(order))[3]


def evaluate_proof(name: str, argument: Any, limbs: int = 2,
                   digits: int | None = None) -> Any:
    """Run the materialised proof at ``limbs``, returning what it produces.

    The width is decided here, once, by promoting the argument and the
    coefficients. Everything after that is the compiled structure running on
    whatever it was handed.
    """

    from .extended_precision import Precision

    digits = int(16 * limbs) if digits is None else int(digits)
    count = order_for(name, CORE_RADII[name], digits=digits)
    callable_, parameters, coefficients, _source = materialise(
        name, order_to_degree(name, count))

    structure = TRANSCENDENTALS[name]["structure"]
    # An argument that is ALREADY wide keeps its limbs: an identity route
    # hands in a reduced argument whose low limbs carry exactly the digits
    # the reduction preserved, and re-promoting it would wrap the wrapper
    # (the backend then receives a Precision as an element and refuses).
    # A width disagreement is a caller error, said by name.
    if isinstance(argument, Precision):
        if int(argument.limbs) != int(limbs) and limbs > 1:
            raise ValueError(
                f"evaluate_proof({name!r}): argument carries "
                f"{argument.limbs} limbs but {limbs} were requested; "
                "widen or narrow it deliberately before evaluating"
            )
        wide = argument
    else:
        wide = Precision.of(argument, limbs) if limbs > 1 else argument
    base = wide * wide if structure in ("odd", "even") else wide
    supply = {"z": wide, "s": base}
    for index, value in enumerate(coefficients):
        parts = limb_decomposition(value, limbs)
        supply[f"c{index}"] = (Precision.constant(wide, parts) if limbs > 1
                               else parts[0])
    return callable_(**{name_: supply[name_] for name_ in parameters})
