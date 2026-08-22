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


def limb_decomposition(rational: Any, limbs: int) -> tuple[float, ...]:
    """An exact rational as ``limbs`` float64 pieces that sum to it.

    ``Fraction`` is exact and ``float(Fraction)`` is correctly rounded, so
    each step takes the nearest double and carries the remainder forward
    without loss. This is the whole reason no arbitrary-precision library is
    needed at build time: the coefficients were rational all along.
    """

    rest = Fraction(int(sympy.numer(rational)), int(sympy.denom(rational)))
    parts = []
    for _ in range(max(int(limbs), 1)):
        head = float(rest)
        parts.append(head)
        rest = rest - Fraction(head)
    return tuple(parts)


def _rational(value: Any) -> Fraction:
    return Fraction(int(sympy.numer(value)), int(sympy.denom(value)))


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


@dataclass(frozen=True)
class SymbolicProgram:
    """One core as compiled structure plus the numbers it takes."""

    name: str
    order: int
    structure: str | None
    #: Parameter names the materialised function expects, in call order.
    parameters: tuple[str, ...]
    #: The exact rational coefficients, in the order the parameters name them.
    coefficients: tuple
    source: str
    callable: Any

    def supply(self, argument: Any, limbs: int) -> dict:
        """Bind the argument and coefficients for a call at ``limbs`` width."""

        from . import extended_precision as xp

        # An odd/even core is a polynomial in z**2 -- that is what carries the
        # parity. A factored or unstructured core is a polynomial in z itself,
        # and squaring its variable would silently evaluate the wrong function.
        squared = self.structure in ("odd", "even")
        base = argument * argument if squared else argument
        bound = {"z": argument, "s": base}
        for index, coefficient in enumerate(self.coefficients):
            bound[f"c{index}"] = xp.constant_limbs(
                argument, limb_decomposition(coefficient, limbs)
            )
        return {name: bound[name] for name in self.parameters}


def _horner(count: int) -> sympy.Expr:
    """Horner over SYMBOLIC coefficients in the structural variable."""

    square = sympy.Symbol("s")
    names = [sympy.Symbol(f"c{index}") for index in range(count)]
    expression = names[-1]
    for symbol in reversed(names[:-1]):
        expression = symbol + square * expression
    return expression


def compile_core(name: str, order: int) -> SymbolicProgram:
    """Derive, structure, and compile one core to AbstractTensor Python.

    The coefficients enter as SYMBOLS and stay symbols all the way through the
    compiler, so the emitted source holds no float literal and the same
    artefact serves every precision.
    """

    from ...compiler.symbolic_equation_compiler import compile_sympy_equations
    from ...compiler.ssa_python_materializer import materialize_function_body

    coefficients = structured_coefficients(name, order)
    structure = TRANSCENDENTALS[name]["structure"]
    body_expression = _horner(len(coefficients))
    if structure in ("odd", "factored"):
        body_expression = sympy.Symbol("z") * body_expression

    compiled = compile_sympy_equations(
        [sympy.Eq(sympy.Symbol("y"), body_expression)],
        name=f"{name}_core",
    )
    statements, needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    if needs_math:
        raise RuntimeError(
            f"{name}: materialised body wants the math module, which means a "
            f"scalar opcode reached a tensor program"
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
    source = ast.unparse(module)
    namespace: dict = {}
    exec(compile(module, f"<{name}_core>", "exec"), namespace)
    return SymbolicProgram(
        name=name, order=order, structure=structure, parameters=parameters,
        coefficients=coefficients, source=source,
        callable=namespace[f"{name}_core"],
    )


def evaluate(program: SymbolicProgram, argument: Any, limbs: int = 2) -> Any:
    """Run a compiled core at ``limbs`` width and collapse to one value."""

    from . import extended_precision as xp

    with xp.precision(limbs):
        promoted = argument + 0.0
        result = program.callable(**program.supply(promoted, limbs))
    return xp.collapse(result)


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
    "ln10": "ln2 * 10/3 corrected, via 2*atanh(9/11) + 2*ln2",
}


def constant_rational(name: str, digits: int = 64) -> Fraction:
    """A transcendental constant as an exact rational, to ``digits``.

    Nothing about this is prepacked: the series came from the identity table,
    the arithmetic is ``Fraction``, and the only approximation is a truncation
    whose order is derived from the requested digits. Ask for more digits and
    the same code produces them -- which is the difference between a constant
    that CONVERGES and a constant that was typed in.
    """

    order = int(digits * 1.6) + 24
    if name == "pi":
        return (16 * _atan_rational(Fraction(1, 5), order)
                - 4 * _atan_rational(Fraction(1, 239), order))
    if name == "e":
        coefficients = structured_coefficients("exp", order)
        return _horner_fraction(coefficients, Fraction(1))
    if name == "ln2":
        return 2 * _atanh_rational(Fraction(1, 3), order)
    if name == "ln10":
        return (2 * _atanh_rational(Fraction(9, 11), order)
                + 2 * constant_rational("ln2", digits))
    raise KeyError(f"no derivation for constant {name!r}")


def constant_limbs(name: str, limbs: int = 2, scale: Fraction | None = None
                   ) -> tuple[float, ...]:
    """A constant as ``limbs`` float64 pieces that sum to it.

    ``scale`` multiplies exactly before decomposition, so ``pi/2`` and
    ``2/pi`` are as exact as ``pi`` -- which matters because argument
    reduction wants those, not pi itself, and forming them from a rounded pi
    would throw away the digits this went to the trouble of deriving.
    """

    digits = int(limbs * 15.95) + 12
    value = constant_rational(name, digits)
    if scale is not None:
        value = value * Fraction(scale)
    rest, parts = value, []
    for _ in range(max(int(limbs), 1)):
        head = float(rest)
        parts.append(head)
        rest = rest - Fraction(head)
    return tuple(parts)


# --------------------------------------------------------------------------
# Exact evaluation: the reference every measurement needs


def reference_program(name: str, radius: float, digits: int = 40):
    """A high-accuracy evaluator that IS the compiled program, run wider.

    There is exactly one implementation of every function here: the one SymPy
    derived and the compiler turned into AbstractTensor Python. A reference
    written any other way -- an arbitrary-precision library, a rational
    re-implementation of the same series -- is a SECOND implementation, and
    then a disagreement between them names no culprit.

    So the reference is the same program at a longer order and more limbs.
    Comparing a shipping configuration against a wider one measures exactly
    what shipping costs: truncation and arithmetic width, the two things the
    configuration chose. Whether the IDENTITY itself is right is a separate
    question, and identities are what answer it -- a wrong identity has to
    survive round trips through unrelated compositions, which an error would
    have to be elaborately harmonised to do.

    Returns a callable taking a tensor and giving back the extended result
    with every limb intact.
    """

    from . import extended_precision as xp

    count = order_for(name, max(abs(float(radius)), 1e-9), digits=digits)
    program = compile_core(name, order_to_degree(name, count))
    limbs = xp.limbs_for_digits(digits)

    def evaluate_reference(argument: Any) -> Any:
        with xp.precision(limbs):
            promoted = argument + 0.0
            return program.callable(**program.supply(promoted, limbs))

    return evaluate_reference


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


def build(name: str, preset: str | Preset | None = None) -> SymbolicProgram:
    """Compile one core at a preset, order derived from the preset's target.

    The default is ``ulp_match``: the fewest limbs that return the
    correctly-rounded double for THIS core, searched rather than declared.
    Defaulting to a named width would be a guess about every core at once,
    and the search costs seconds at bake time to remove it.
    """

    if preset is None:
        chosen = ulp_matched(name)
    else:
        chosen = preset if isinstance(preset, Preset) else PRESETS[str(preset)]
    radius = CORE_RADII.get(name, 1.0)
    count = order_for(name, radius, digits=chosen.digits)
    return compile_core(name, order_to_degree(name, count))


def run(program: SymbolicProgram, argument: Any,
        preset: str | Preset | None = None) -> Any:
    """Evaluate a compiled core at a preset's arithmetic width."""

    from . import extended_precision as xp

    if preset is None:
        chosen = ulp_matched(program.name)
    else:
        chosen = preset if isinstance(preset, Preset) else PRESETS[str(preset)]
    with xp.precision(chosen.limbs):
        promoted = argument + 0.0
        result = program.callable(**program.supply(promoted, chosen.limbs))
    settled_value = xp.collapse(result)
    if not chosen.rounding_test:
        return settled_value

    # Ziv, in its cheap form: recompute one limb wider and see whether the
    # double answer moves. If it does not, the result was already settled and
    # no wider evaluation can change it. This DETECTS the unsettled cases; it
    # does not prove none exist, which needs the hardest-to-round bounds.
    wider = chosen.limbs + 1
    with xp.precision(wider):
        promoted = argument + 0.0
        again = xp.collapse(program.callable(**program.supply(promoted, wider)))
    return again


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
DIGITS_PER_LIMB_ESTIMATE = 15.95


def ulp_matched(name: str, radius: float | None = None,
                samples: int = 401, ceiling: int = 6) -> Preset:
    """The FEWEST limbs that return the correctly-rounded double, per core.

    A global limb count is a guess dressed as a policy: too wide for one core,
    too narrow for another, and nothing about a core's identity says which it
    is. So it is searched per core against the independent oracle, and the
    answer is the first width that matches every sampled point.

    The order is not searched alongside it. Each width has an arithmetic floor
    near ``1e-16*limbs``, and sizing truncation below the floor the arithmetic
    already imposes buys nothing -- so the order FOLLOWS from the width, and
    the width is the only thing that moves.

    Matching on a sample is an OBSERVATION, not a proof: a tie can still fall
    the wrong way somewhere unsampled. ``exact_preset`` is the stronger claim.
    """

    import numpy as _np

    from . import extended_precision as xp
    from .abstraction import AbstractTensor

    radius = float(CORE_RADII[name] if radius is None else radius)
    points = _np.linspace(-radius * 0.98, radius * 0.98, int(samples))
    points = points[_np.abs(points) > 1e-12]
    oracle = exact_evaluator(name, radius, digits=40)
    truth = _np.array([float(oracle(float(value))) for value in points])
    tensor = AbstractTensor.get_tensor(points)

    for limbs in range(1, int(ceiling) + 1):
        digits = int(DIGITS_PER_LIMB_ESTIMATE * limbs)
        count = order_for(name, radius, digits=digits)
        program = compile_core(name, order_to_degree(name, count))
        with xp.precision(limbs):
            promoted = tensor + 0.0
            produced = program.callable(**program.supply(promoted, limbs))
        got = _np.asarray(xp.collapse(produced).tolist(), dtype=float).ravel()
        if _np.array_equal(got, truth):
            return Preset(
                f"ulp_match:{name}", digits=digits, limbs=limbs,
                note=f"{count} coefficients, matched on {points.size} points",
            )
    raise ValueError(
        f"{name}: no width up to {ceiling} limbs matched the oracle; the "
        f"interval or the identity is at fault, not the arithmetic"
    )


def exact_preset(name: str, radius: float | None = None) -> Preset:
    """Correct rounding as a SEARCH, not a sampled observation.

    ``ulp_matched`` reports a width that agreed everywhere it looked. This
    keeps widening until the double answer stops moving -- Ziv's criterion.
    Once a wider evaluation cannot change the result, the result is settled
    and further width buys literally nothing. That is the
    return-on-investment limit.
    """

    base = ulp_matched(name, radius)
    return Preset(f"exact:{name}", digits=base.digits + 16,
                  limbs=base.limbs + 1, rounding_test=True,
                  note="widened past the matched width and verified settled")
