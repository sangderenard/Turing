"""Signal-math trigonometry authored as COMPILER SOURCE, the way BLAS is.

``signal_math`` owns the mathematics: which reduction, which polynomial form,
which coefficients, and what residual they measure. This module turns one
baked core set into the artefact the kernel bank consumes -- a plain-Python
source string with the loop bound as a PARAMETER -- so trigonometry becomes a
pack in the same sense ``blas.py`` is one, rather than a single frozen
capture with a fallback.

Why source and not a captured graph
-----------------------------------
An SSA capture fixes its shape: ``SSATensorProgram.input`` takes a literal
``tuple[int, ...]``, so every extent folds to a constant before it reaches the
emitter. A pack baked that way answers exactly the width it was baked at and
nothing else -- measured, it served 1 call in 4 and sent the rest back to the
interpreter, which is slower than the libm operator it replaced.

Authoring the same mathematics as source with ``range(n)`` gives the kernel
bank a genuinely PARAMETRIC variant plus whatever size-specialized ones a
matrix asks for, and ``LaunchCoordinator`` then routes per call exactly as it
does for ``gemm``. That is the whole difference between a baked artefact and
a pack -- and it applies to the REVERSE as much as the forward.

The authoring rules from ``blas.py`` hold here and are not restated: flat 1-D
buffers with computed indices, every loop bound a passed-in parameter, loop
variables named distinctly, and no hand optimization -- the compiler's own
pipeline is what turns this into a fast kernel.

The oracle cannot drift
-----------------------
:func:`kernel_reference` obtains the reference by EXECUTING the same source
string it hands the compiler. The two therefore cannot disagree about what
the kernel means, which keeps bank admission answering the question it is
for -- *did the compiler compile this faithfully?* -- separate from the
question ``signal_math`` already answers with its own measurement, *does this
polynomial approximate the function to epsilon?*
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping

import numpy as np

from . import signal_math as _signal

#: What the kernels bake at unless a caller says otherwise.
#: The working type is float64, so "at least double precision"
#: means the result should be right to the last bit or two of one
#: -- about 2.22e-16 relative, one ulp. The looser settings exist
#: for footprint, not for speed: measured on the compiled sine,
#: going from 5 coefficients to 7 bought roughly 36,000x accuracy
#: for about 10% more time, because the octant reduction and the
#: memory traffic dominate, not the polynomial. There is no
#: accuracy/speed trade here worth taking, so the default takes
#: the accuracy.
DEFAULT_KERNEL_QUALITY = "double"


def _literal(value: float) -> str:
    """Round-trippable decimal for a coefficient baked into source."""

    return repr(float(value))


def _horner_expression(variable: str, coefficients: tuple[float, ...]) -> str:
    """Nested Horner as source text, innermost coefficient first."""

    expression = _literal(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        expression = f"({_literal(coefficient)} + {variable} * {expression})"
    return expression


def core_expression(core: Any, argument: str, square: str) -> str:
    """One core's evaluation as source, in whatever form it was baked.

    The selector picks a family per core on measured evidence, so a kernel
    cannot assume one. ``exact`` and ``structured`` share a form -- the parity
    is in the expression, evaluated in the SQUARE of the argument -- and
    differ only in where their coefficients came from. ``series`` is a plain
    Horner in the argument itself. Emitting the wrong one silently computes a
    different function, so this refuses anything it does not know rather than
    guessing a form.
    """

    if core.family in ("exact", "structured"):
        polynomial = _horner_expression(square, core.values)
        if core.structure == "odd":
            return f"{argument} * {polynomial}"
        if core.structure == "even":
            return polynomial
        raise ValueError(
            f"{core.core} is {core.structure!r}; the circular kernel emits "
            f"only odd and even forms"
        )
    if core.family == "series":
        shifted = (
            argument if not core.centre
            else f"({argument} - {_literal(core.centre)})"
        )
        return _horner_expression(shifted, core.values)
    raise ValueError(
        f"{core.core} was baked as {core.family!r}, which this kernel cannot "
        f"emit; bake it exact, structured or series"
    )


def _reduction_lines(cores: _signal.CoreSet, phase: str) -> str:
    """The octant reduction and both polynomials, as loop-body source.

    Shared verbatim by the forward and the VJP so the two cannot drift in
    their reduction -- which is where the accuracy actually lives.

    Both cores are evaluated every iteration and the octant selects between
    them: each core serves the octant where the OTHER has its zero, so neither
    polynomial is ever asked to be accurate across its own root.
    """

    sine, cosine = cores["sin"], cores["cos"]
    return (
        f"        v = x[i]{phase}\n"
        f"        t = v * {_literal(1.0 / math.tau)}\n"
        f"        s = t * 4.0 + 0.5\n"
        f"        k = s - (s % 1.0)\n"
        f"        r = (t - k * 0.25) * {_literal(math.tau)}\n"
        f"        w = k * 0.25\n"
        f"        q = k - (w - (w % 1.0)) * 4.0\n"
        f"        u = r * r\n"
        f"        sp = {core_expression(sine, 'r', 'u')}\n"
        f"        cp = {core_expression(cosine, 'r', 'u')}\n"
    )


def _selection_lines(target: str, factor: str) -> str:
    """Store the octant-selected value, scaled by ``factor``."""

    return (
        f"        if q < 1.0:\n"
        f"            {target} = {factor}sp\n"
        f"        elif q < 2.0:\n"
        f"            {target} = {factor}cp\n"
        f"        elif q < 3.0:\n"
        f"            {target} = -{factor}sp\n"
        f"        else:\n"
        f"            {target} = -{factor}cp\n"
    )


def circular_kernel_source(cores: _signal.CoreSet, name: str = "sin") -> str:
    """Elementwise ``sin``/``cos`` over radians, octant-reduced."""

    if name not in ("sin", "cos"):
        raise ValueError(f"circular kernel must be sin or cos, got {name!r}")
    # cos(v) is sin(v + tau/4); the shift is exact and keeps one code path.
    phase = "" if name == "sin" else f" + {_literal(math.tau / 4.0)}"
    return (
        f"\ndef {name}(x, y, n):\n"
        f"    for i in range(n):\n"
        + _reduction_lines(cores, phase)
        + _selection_lines("y[i]", "")
        + "    return y\n"
    )


def exp_kernel_source(core: Any) -> str:
    """``exp`` by ``2**k * exp(r)``, with ``r`` inside the baked band.

    Uses the SERIES family deliberately. A polyspline core would need a
    segment-selection chain per element, and the exponential's series has
    exact rational coefficients and converges fast on a half-ln2 band -- so
    the series is both the cheaper and the more accurate choice here, which
    is not the usual ordering and is worth stating.

    ``k`` is obtained by ``s - (s % 1.0)``: the compiled ``%`` follows
    Python's floored semantics, measured, so that IS floor rather than
    truncation.
    """


    return (
        "\ndef exp(x, y, n):\n"
        "    for i in range(n):\n"
        f"        s = x[i] * {_literal(1.0 / math.log(2.0))} + 0.5\n"
        "        k = s - (s % 1.0)\n"
        f"        r = x[i] - k * {_literal(math.log(2.0))}\n"
        f"        y[i] = {core_expression(core, 'r', 'r * r')} * (2.0 ** k)\n"
        "    return y\n"
    )


#: The backward rules this module can author: those whose derivative is a
#: signed multiple of a PARTNER function of the same argument.
_PARTNER_RULE = re.compile(
    r"^gx = unbroadcast\((-?)g \* (\w+)\(x\), x\.shape\)$"
)

#: Derivatives expressed in the method's OWN OUTPUT rather than its argument:
#: ``tan`` -> ``g * (1 + y*y)``, ``tanh`` -> ``g * (1 - y*y)``. The forward
#: value is already computed by the reduction, so these cost one extra
#: multiply-add and no second evaluation.
_OUTPUT_RULE = re.compile(
    r"^gx = unbroadcast\(g \* \(1 ([+-]) .*\), x\.shape\)$"
)


#: Derivatives that are a signed reciprocal of a QUADRATIC in the argument:
#: ``atan`` -> ``g / (1 + x*x)``, ``atanh`` -> ``g / (1 - x*x)``. These need
#: no core at all -- they are arithmetic -- which is why they land before
#: their own forwards do.
_QUOTIENT_RULE = re.compile(
    r"^gx = unbroadcast\((-?)g / \(1 ([+-]) x\*x\), x\.shape\)$"
)

#: The same shape but through a square root: ``asin``, ``acos``, ``asinh``,
#: ``acosh``. Blocked, and not on the derivative -- their radicands are
#: ``1-x*x`` on [0,1], ``x*x+1`` unbounded above and ``x*x-1`` from zero, none
#: of which sit in the mantissa band the sqrt kernel takes. Inlining sqrt
#: needs in-kernel range reduction, which is the same ``frexp`` gap the sqrt
#: kernel itself is bounded by.
_ROOT_QUOTIENT_RULE = re.compile(
    r"^gx = unbroadcast\((-?)g / sqrt\((.+)\), x\.shape\)$"
)


def vjp_plan(name: str) -> tuple[str, str]:
    """How ``name``'s reverse should be obtained, and why.

    Four answers, and the FIRST is the default and the most important:

    ``identity``
        The registry declares no backward rule for this method, because the
        authored surface composes it from others -- ``sec`` is ``cos() ** -1``.
        Differentiating that composition is already correct and already
        happens. Baking a dedicated reverse here would invent a rule the
        registry deliberately withholds, and a second statement of a
        derivative is a second chance to disagree with the first. So anything
        reasonable as an identity is BYPASSED, not baked: no kernel, no
        variant matrix row, nothing to keep in step.

    ``partner`` / ``output``
        A rule exists and has a shape this module can author.

    ``unsupported``
        A rule exists in a shape this module will not guess at.
    """

    from .backward_registry import BACKWARD_RULES

    entry = BACKWARD_RULES.get(str(name)) or {}
    rule = (entry.get("backward") or {}).get("x")
    if rule is None:
        return "identity", (
            "no backward rule is declared; the authored surface composes this "
            "method from others and differentiating that composition is the "
            "derivative"
        )
    text = str(rule).strip()
    if _PARTNER_RULE.match(text):
        return "partner", text
    if _OUTPUT_RULE.match(text):
        return "output", text
    if _QUOTIENT_RULE.match(text):
        return "quotient", text
    if _ROOT_QUOTIENT_RULE.match(text):
        return "root_quotient", text
    return "unsupported", text


def _octant_chain(target: str, branches: tuple[str, str, str, str]) -> str:
    """One if/elif chain writing ONE statement per branch.

    ``target`` empty means each branch entry is already a full statement;
    otherwise it is assigned the branch expression. Deliberately one
    statement per branch, and it must WRITE ITS DESTINATION here rather than
    set a local for use after the merge: the latter emits a pointer-valued
    phi that does not dominate its uses, LLVM rejects the module, and
    ``artifact.shortfalls`` stays empty -- the false green ``blas.py``
    records under section 4.1b.
    """

    statements = tuple(
        branch if not target else f"{target} = {branch}" for branch in branches
    )
    first, second, third, fourth = statements
    return (
        f"        if q < 1.0:\n            {first}\n"
        f"        elif q < 2.0:\n            {second}\n"
        f"        elif q < 3.0:\n            {third}\n"
        f"        else:\n            {fourth}\n"
    )


#: ``tan`` per octant. The sine is ``sp, cp, -sp, -cp`` and the cosine is
#: ``cp, -sp, -cp, sp``, so the ratio alternates and the signs cancel in
#: pairs.
_TANGENT_BRANCHES = ("sp / cp", "-cp / sp", "sp / cp", "-cp / sp")


def _tangent_selection_lines(store: Callable[[str], str]) -> str:
    """``tan`` selected and STORED inside each branch.

    The store must happen in the branch. Assigning a local in each arm and
    using it after the merge emits a pointer-valued phi that does not
    dominate its uses, and LLVM rejects the module while
    ``artifact.shortfalls`` stays empty -- the false green ``blas.py``
    records under section 4.1b. Writing the destination inside the branch is
    what the working ``sin``/``cos`` kernels do, and it is an authoring
    constraint here for the same reason distinct loop variable names are.

    ``store`` receives the branch's expression and returns the statement,
    so the forward and the VJP share the selection and differ only in what
    they write.
    """

    return _octant_chain(
        "", tuple(store(branch) for branch in _TANGENT_BRANCHES),
    )


def tan_kernel_source(cores: _signal.CoreSet) -> str:
    """``tan`` as the ratio of one octant reduction's two polynomials."""

    return (
        "\ndef tan(x, y, n):\n"
        "    for i in range(n):\n"
        + _reduction_lines(cores, "")
        + _tangent_selection_lines(lambda ratio: f"y[i] = {ratio}")
        + "    return y\n"
    )


def output_vjp_source(cores: _signal.CoreSet, name: str) -> str:
    """A VJP stated in the method's OWN output, ``g * (1 +- y*y)``.

    The forward value is recovered from the same reduction the forward uses,
    so the derivative costs one multiply-add rather than a second evaluation.
    The sign is taken from the authored rule, never assumed here.
    """

    plan, rule = vjp_plan(name)
    if plan != "output":
        raise ValueError(f"{name} is not an output-form rule: {plan} ({rule})")
    if name != "tan":
        raise ValueError(
            f"{name} needs a hyperbolic core to recover its forward value; "
            f"only the circular reduction is authored here"
        )
    sign = _OUTPUT_RULE.match(rule).group(1)
    return (
        f"\ndef {name}_vjp(x, g, d, n):\n"
        "    for i in range(n):\n"
        + _reduction_lines(cores, "")
        + _tangent_selection_lines(
            lambda ratio: f"d[i] = g[i] * (1.0 {sign} ({ratio}) * ({ratio}))"
        )
        + "    return d\n"
    )


def circular_vjp_source(cores: _signal.CoreSet, name: str) -> str:
    """The VJP of one circular method, DERIVED from its authored rule.

    Not a restatement of the derivative. ``BACKWARD_RULES`` already owns what
    d/dx of each method is; this reads that entry and emits a kernel only when
    the rule has the shape ``g * partner(x)`` with an optional sign -- exactly
    the case for the circular pair. Any other shape raises, because the
    alternative is a second author's opinion about a derivative, silently
    disagreeing with the first.

    The reduction is the forward's, verbatim, so the VJP is parametric for the
    same reason the forward is: ``n`` is a real loop bound. That is what stops
    a reverse from being frozen at the width it was captured at.
    """

    from .backward_registry import BACKWARD_RULES

    rule = ((BACKWARD_RULES.get(name) or {}).get("backward") or {}).get("x")
    if rule is None:
        raise ValueError(f"{name} declares no backward rule for x")
    match = _PARTNER_RULE.match(str(rule).strip())
    if match is None:
        raise ValueError(
            f"{name} backward rule is not a signed partner function, so this "
            f"module will not author its VJP: {rule!r}"
        )
    sign, partner = match.group(1), match.group(2)
    if partner not in ("sin", "cos"):
        raise ValueError(
            f"{name} differentiates to {partner!r}, which has no baked "
            f"circular core here"
        )
    # The derivative of `name` IS the partner, so the partner's own phase is
    # what the reduction runs with.
    phase = "" if partner == "sin" else f" + {_literal(math.tau / 4.0)}"
    factor = f"{sign}g[i] * "
    return (
        f"\ndef {name}_vjp(x, g, d, n):\n"
        f"    for i in range(n):\n"
        + _reduction_lines(cores, phase)
        + _selection_lines("d[i]", factor)
        + "    return d\n"
    )




#: Newton's iteration for sqrt is a fixed point of the function itself,
#: ``y <- (y + x/y)/2``, so it is SELF-CORRECTING: the seed's accuracy barely
#: matters and each step roughly squares the correct digits. Measured on the
#: mantissa band, that is worth far more than a better polynomial:
#:
#:     series core, 48 coefficients          328.10 ulp p95
#:     the same core plus one Newton step      0.79 ulp p95
#:     degree-6 seed plus two Newton steps     0.79 ulp p95
#:
#: Seven coefficients and two steps land where forty-eight coefficients
#: cannot. The rewrite is worth more than the approximation, which is the
#: identity argument in miniature.
SQRT_NEWTON_STEPS = 2


def sqrt_seed(degree: int = 6) -> tuple[float, ...]:
    """A cheap relative-weighted polynomial seed for the mantissa band."""

    import mpmath

    nodes = np.linspace(0.25, 1.0, 8 * (int(degree) + 1))
    with mpmath.workdps(40):
        values = np.array(
            [float(mpmath.sqrt(mpmath.mpf(float(node)))) for node in nodes]
        )
    coefficients = np.polynomial.polynomial.polyfit(
        nodes, values, int(degree), w=1.0 / np.abs(values),
    )
    return tuple(float(value) for value in coefficients)


def sqrt_kernel_source(seed: tuple[float, ...],
                       steps: int = SQRT_NEWTON_STEPS) -> str:
    """``sqrt`` on the MANTISSA BAND, seeded then Newton-refined.

    Takes an argument already reduced to ``[0.25, 1)``; the caller supplies
    the even binade, exactly as the angle palette takes an index. That
    boundary is deliberate and currently forced: extracting the exponent
    inside the kernel needs either a ``frexp`` primitive the authored
    vocabulary lacks, or a data-dependent ``while`` loop -- and a probe of
    the latter HUNG the compiler for ten minutes without producing a module,
    so it is not an option today.

    The caller's reduction is free: scaling by a power of four is exact, so
    ``sqrt(m * 4**k) = 2**k * sqrt(m)`` holds to the bit.
    """

    body = [
        "\ndef sqrt(x, y, n):\n",
        "    for i in range(n):\n",
        "        m = x[i]\n",
        f"        r = {_horner_expression('m', seed)}\n",
    ]
    body.extend("        r = 0.5 * (r + m / r)\n" for _ in range(int(steps)))
    body.append("        y[i] = r\n")
    body.append("    return y\n")
    return "".join(body)


def quotient_vjp_source(name: str) -> str:
    """A VJP that is a signed reciprocal of a quadratic -- pure arithmetic.

    ``atan`` and ``atanh`` differentiate to ``g / (1 +- x*x)``, which needs no
    core, no reduction and no table. They are therefore authorable before
    their own forwards are, and their accuracy is whatever the division and
    one multiply-add give -- which is why the sign and the operator are taken
    from the authored rule rather than written out here: the only thing this
    function contributes is the loop.
    """

    plan, rule = vjp_plan(name)
    if plan != "quotient":
        raise ValueError(f"{name} is not a quadratic-quotient rule: {plan}")
    sign, operator = _QUOTIENT_RULE.match(rule).groups()
    return (
        f"\ndef {name}_vjp(x, g, d, n):\n"
        "    for i in range(n):\n"
        "        v = x[i]\n"
        f"        d[i] = {sign}g[i] / (1.0 {operator} v * v)\n"
        "    return d\n"
    )


def kernel_reference(source: str, name: str) -> Callable[..., Any]:
    """The oracle: the same source, executed as Python.

    Deliberately not a second implementation. A hand-written twin would be
    answering "do two authors agree", which is not the question bank
    admission asks.
    """

    namespace: dict[str, Any] = {}
    exec(compile(source, f"<signal-kernel {name}>", "exec"), namespace)
    return namespace[name]


#: Forwards this module authors, and how each is emitted. ``tan`` shares the
#: circular reduction rather than getting its own, so all three cost one
#: octant reduction and differ only in what the branch stores.
_FORWARD_SOURCE: Mapping[str, Callable[[Any], str]] = {
    "sin": lambda cores: circular_kernel_source(cores, "sin"),
    "cos": lambda cores: circular_kernel_source(cores, "cos"),
    "tan": tan_kernel_source,
    "exp": lambda cores: exp_kernel_source(cores["exp"]),
}


def kernel_spec(cores: _signal.CoreSet, name: str):
    """One :class:`KernelSpec` for the forward of ``name``."""

    from ...compiler.kernel_bank import KernelSpec

    source = _FORWARD_SOURCE[name](cores)
    return KernelSpec(
        name=name, source=source, function_name=name,
        reference=kernel_reference(source, name),
        parameter_order=("x", "y", "n"), size_parameters=("n",),
        example_inputs=lambda sizes, rng: {
            "x": rng.uniform(-8.0, 8.0, sizes["n"]),
            "y": np.zeros(sizes["n"]), "n": int(sizes["n"]),
        },
        extents={"x": ("n",), "y": ("n",)},
    )


def vjp_spec(cores: _signal.CoreSet, name: str):
    """One :class:`KernelSpec` for the parametric VJP of ``name``."""

    from ...compiler.kernel_bank import KernelSpec

    plan, rule = vjp_plan(name)
    if plan == "identity":
        raise ValueError(
            f"{name} is an identity over other methods and is bypassed, not "
            f"baked: {rule}"
        )
    if plan == "quotient":
        source = quotient_vjp_source(name)
    elif plan == "output":
        source = output_vjp_source(cores, name)
    else:
        source = circular_vjp_source(cores, name)
    return KernelSpec(
        name=f"{name}_vjp", source=source, function_name=f"{name}_vjp",
        reference=kernel_reference(source, f"{name}_vjp"),
        parameter_order=("x", "g", "d", "n"), size_parameters=("n",),
        example_inputs=lambda sizes, rng: {
            "x": rng.uniform(-8.0, 8.0, sizes["n"]),
            "g": rng.standard_normal(sizes["n"]),
            "d": np.zeros(sizes["n"]), "n": int(sizes["n"]),
        },
        extents={"x": ("n",), "g": ("n",), "d": ("n",)},
    )


def signal_kernel_specs(quality: str = DEFAULT_KERNEL_QUALITY, *,
                        include_vjp: bool = True) -> Mapping[str, Any]:
    """Every kernel this module authors, at one baked quality.

    Only what has to be baked. Methods whose derivative is an identity over
    these -- ``sec``, ``csc``, ``cot``, ``sech``, ``csch``, ``coth``,
    ``sinc`` -- are absent on purpose: see :func:`vjp_plan`. Skipping them is
    not a coverage gap, it is what keeps the variant matrix from multiplying
    by methods that have nothing of their own to compute.
    """

    cores = _signal.signal_math(quality).cores
    specs = {name: kernel_spec(cores, name) for name in _FORWARD_SOURCE}
    if include_vjp:
        for name in _FORWARD_SOURCE:
            # Only the plans this module can author. `identity` is bypassed by
            # design; `unsupported` is a rule shape it will not guess at.
            if vjp_plan(name)[0] not in ("partner", "output", "quotient"):
                continue
            try:
                specs[f"{name}_vjp"] = vjp_spec(cores, name)
            except ValueError:
                continue
    return specs


__all__ = [
    "quotient_vjp_source",
    "SQRT_NEWTON_STEPS",
    "sqrt_kernel_source",
    "sqrt_seed",
    "core_expression",
    "DEFAULT_KERNEL_QUALITY",
    "output_vjp_source",
    "tan_kernel_source",
    "vjp_plan",
    "circular_kernel_source",
    "circular_vjp_source",
    "kernel_reference",
    "kernel_spec",
    "signal_kernel_specs",
    "vjp_spec",
]
