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


_TANGENT_BRANCHES = ("sp / cp", "-cp / sp", "sp / cp", "-cp / sp")


SQRT_NEWTON_STEPS = 2


def kernel_reference(source: str, name: str) -> Callable[..., Any]:
    """The oracle: the same source, executed as Python.

    Deliberately not a second implementation. A hand-written twin would be
    answering "do two authors agree", which is not the question bank
    admission asks.
    """

    namespace: dict[str, Any] = {}
    exec(compile(source, f"<signal-kernel {name}>", "exec"), namespace)
    return namespace[name]


#: Forwards this module bakes. What each COMPUTES is not stated here -- it
#: comes from the identity table, through SymPy, through the compiler.
FORWARD_KERNELS: tuple[str, ...] = ("sin", "cos", "tan", "exp")

#: The kernel's shape, authored once as ordinary Python rather than assembled
#: from fragments. The loop and the indexing are structure, not mathematics,
#: so they are written out; the mathematics arrives as ``core`` beneath, and
#: nothing here knows what function that is.
#: The coefficients arrive as ONE BUFFER, not as scalars and never as
#: literals. As scalars the compiler did not expose them as ports -- the
#: emitted entry carried only x, y and n -- and as literals they would be
#: ``repr(float(c))``, which is where a derived coefficient loses every limb
#: past the first. A buffer is unambiguously data the caller supplies.
KERNEL_SHELL = """
def {name}(x{annotate}, y{annotate}, n, c{annotate}):
    for i in range(n):
        z = x[i]
        s = {structural}
        y[i] = {name}_core({arguments})
    return y
"""


def _materialised_core(name: str, digits: int) -> tuple[str, tuple[str, ...]]:
    """The proof, compiled to AbstractTensor Python, as source.

    This module used to build its kernels by pasting coefficients into
    f-strings. That is a second implementation of the mathematics, and its
    smallest unit was ``repr(float(c))`` -- ONE double -- so a coefficient
    derived to four limbs lost three of them at emission and no amount of
    width downstream could recover them.

    Nothing is templated now. The identity goes to SymPy, the reduced series
    to ``compile_sympy_equations``, the SSA to ``materialize_function_body``,
    and what comes back is real AbstractTensor Python that this only has to
    unparse.
    """

    from . import signal_symbolic as _proof

    count = _proof.order_for(name, _proof.CORE_RADII[name], digits=digits)
    order = _proof.order_to_degree(name, count)
    _callable, parameters, _coefficients, source = _proof.materialise(
        name, order)
    return source, parameters


#: The axes a variant can differ on. ``digits`` sizes the ORDER through the
#: exact tail bound, so each entry is a different amount of series and a
#: different accuracy -- not a different spelling of one program. A core times
#: an accuracy is the matrix, and every cell compiles and is verified
#: separately, so a cell that cannot hold its target says so instead of being
#: assumed from its neighbour.
ACCURACY_AXIS: tuple[int, ...] = (8, 17, 24, 32)


def _radius(name: str) -> float:
    """The half-width of a core's own interval."""

    from . import signal_symbolic as _proof

    return float(_proof.CORE_RADII[name])


def _wide_reference(name: str, order: int, width: int,
                    coefficient_names: tuple[str, ...]):
    """The same materialised program, run eagerly at its declared width.

    A width-annotated kernel's semantics ARE the limb expansion, so its
    oracle cannot be the source exec'd over plain buffers -- indexing
    ``x[i]`` there reads one limb slot, not one element. The honest twin is
    the eager Precision path over the SAME materialised callable: interleaved
    buffers wrap directly (interleaving is Precision's own layout), the
    coefficients rebuild from their limb slots, and the produced limbs write
    back interleaved. Still one authorship: identity, SymPy, compiler.
    """

    from .abstraction import AbstractTensor
    from .extended_precision import Precision
    from . import signal_symbolic as _proof

    callable_, parameters, _coefficients, _source = _proof.materialise(
        name, order)
    structure = _proof.TRANSCENDENTALS[name]["structure"]

    parameter_order = (
        "x", "y", "n", *coefficient_names,
        *(
            f"{coefficient}__limb{position}"
            for coefficient in coefficient_names
            for position in range(1, width)
        ),
    )

    def reference(*arguments, **named):
        # The bank calls references POSITIONALLY in parameter_order; eager
        # callers may name them. Both spell the same binding.
        import numpy as np

        named = {**dict(zip(parameter_order, arguments)), **named}
        x, y, n = named.pop("x"), named.pop("y"), named.pop("n")
        count = int(n)
        wide = Precision(
            AbstractTensor.get_tensor(
                np.asarray(x, dtype=np.float64)[: count * width]
            ),
            width,
        )
        base = wide * wide if structure in ("odd", "even") else wide
        supply = {"z": wide, "s": base}
        for coefficient in coefficient_names:
            parts = (float(named[coefficient]), *(
                float(named[f"{coefficient}__limb{position}"])
                for position in range(1, width)
            ))
            supply[coefficient] = Precision.constant(wide, parts)
        produced = callable_(
            **{parameter: supply[parameter] for parameter in parameters}
        )
        flat = np.asarray(
            produced._value.tolist(), dtype=np.float64
        ).ravel()
        y = np.asarray(y, dtype=np.float64)
        y[: count * width] = flat[: count * width]
        return y

    return reference


def kernel_spec(cores: _signal.CoreSet, name: str, digits: int = 32,
                limb_width: int = 1):
    """One :class:`KernelSpec` for the forward of ``name`` at ``digits``.

    ``limb_width`` is the tier's depth, and EVERYTHING in the kernel is at
    that depth: the source carries ``Precision[width]`` annotations so the
    compiler's precision pipeline expands the arithmetic, and the
    coefficient buffer holds each exact rational decomposed to exactly that
    many limbs -- the coefficients of a tier are the depth of the tier.
    """

    from ...compiler.kernel_bank import KernelSpec

    from . import signal_symbolic as _proof

    body, parameters = _materialised_core(name, digits=digits)
    coefficients = tuple(sorted(
        (each for each in parameters if each.startswith("c")),
        key=lambda each: int(each[1:]),
    ))
    structure = _proof.TRANSCENDENTALS[name]["structure"]
    # The structural variable: a parity core is a polynomial in the square,
    # anything else in the argument itself. Squaring the wrong one computes a
    # different function, so it comes from the identity table rather than a
    # guess.
    width = max(int(limb_width), 1)
    annotate = f": Precision[{width}]" if width > 1 else ""
    if width > 1:
        # The WIDE kernel takes the exact authored shape the precision
        # benchmark proves daily: the core's statements INLINED into the
        # loop and the coefficients as annotated scalar parameters, whose
        # low limbs the pipeline appends as extra formals. The buffer-and-
        # call spelling lowers along a different seam that does not yet
        # stride a constant-indexed coefficient array, and a kernel is not
        # the place to prove a new seam -- it is the place to use the
        # proven one.
        statements = [
            line for line in body.splitlines()[1:] if line.strip()
        ]
        returned = statements[-1].split("return")[-1].strip()
        inlined = [("    " + line) for line in statements[:-1]]
        source = "\n".join((
            "",
            f"def {name}(x{annotate}, y{annotate}, n, "
            + ", ".join(f"{c}{annotate}" for c in coefficients) + "):",
            "    for i in range(n):",
            "        z = x[i]",
            "        s = "
            + ("z * z" if structure in ("odd", "even") else "z"),
            *inlined,
            f"        y[i] = {returned}",
            "    return y",
            "",
        ))
    else:
        reads = {each: f"c[{int(each[1:])}]" for each in coefficients}
        shell = KERNEL_SHELL.format(
            name=name,
            annotate=annotate,
            structural="z * z" if structure in ("odd", "even") else "z",
            arguments=", ".join(reads.get(each, each) for each in parameters),
        )
        source = body + shell
    order = _proof.order_to_degree(
        name, _proof.order_for(name, _radius(name), digits=digits))
    exact = _proof.structured_coefficients(name, order)

    if width > 1:
        # The coefficients of a tier are the DEPTH of the tier: each exact
        # rational decomposed to exactly ``width`` limbs. The head keeps
        # the authored name and the tails take ``{name}__limb{j}`` -- the
        # spelling the bank derives for the pipeline's appended formals --
        # so every limb is a named, fed parameter and none is baked or
        # silently zeroed.
        constants: dict[str, float] = {}
        for coefficient, value in zip(coefficients, exact):
            parts = _proof.limb_decomposition(value, width)
            constants[coefficient] = float(parts[0])
            for position in range(1, width):
                constants[f"{coefficient}__limb{position}"] = float(
                    parts[position]
                )

        def example_inputs(sizes, rng, _r=_radius(name), _w=width,
                           _constants=constants):
            count = int(sizes["n"])
            x = np.zeros(count * _w)
            x[::_w] = rng.uniform(-_r * 0.98, _r * 0.98, count)
            return {
                "x": x, "y": np.zeros(count * _w), "n": count,
                **{key: value for key, value in _constants.items()},
            }

        return KernelSpec(
            name=f"{name}_d{digits}", source=source, function_name=name,
            reference=_wide_reference(name, order, width, coefficients),
            parameter_order=(
                "x", "y", "n", *coefficients,
                *(
                    f"{coefficient}__limb{position}"
                    for coefficient in coefficients
                    for position in range(1, width)
                ),
            ),
            size_parameters=("n",),
            example_inputs=example_inputs,
            # Wide buffers hold width limbs per element, which the extent
            # vocabulary cannot yet say; None makes partitioning machinery
            # refuse rather than tear limb pairs across a chunk boundary.
            extents=None,
            limb_width=width,
        )

    values = np.asarray(
        [float(_proof.limb_decomposition(value, 1)[0]) for value in exact],
        dtype=np.float64,
    )

    def example_inputs(sizes, rng, _r=_radius(name), _v=values):
        return {
            "x": rng.uniform(-_r * 0.98, _r * 0.98, sizes["n"]),
            "y": np.zeros(sizes["n"]), "n": int(sizes["n"]),
            "c": _v.copy(),
        }

    return KernelSpec(
        name=f"{name}_d{digits}", source=source, function_name=name,
        reference=kernel_reference(source, name),
        parameter_order=("x", "y", "n", "c"), size_parameters=("n",),
        # Sampled on the CORE'S OWN interval. Verification compares the
        # kernel against its Python reference, and outside the interval both
        # compute the same nonsense and agree -- admission would then mean
        # nothing. A core is only claimed where it is valid.
        example_inputs=example_inputs,
        extents={"x": ("n",), "y": ("n",), "c": ()},
        limb_width=1,
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
    # The tier's depth decides every kernel's width: a quality whose epsilon
    # lies past one double's resolution builds width-annotated kernels with
    # limb-decomposed coefficients, so the compiled artifact IS the tier
    # rather than a single-double impression of it.
    import math as _math

    from .extended_precision import limbs_for_digits as _limbs_for_digits
    from .signal_math import PREBAKE_SETS as _sets

    epsilon = float(_sets[str(quality)].epsilon)
    width = _limbs_for_digits(int(_math.ceil(-_math.log10(epsilon))))
    specs = {}
    for name in FORWARD_KERNELS:
        spec = kernel_spec(cores, name, limb_width=int(width))
        specs[spec.name] = spec
    if include_vjp and False:
        for name in FORWARD_KERNELS:
            # Only the plans this module can author. `identity` is bypassed by
            # design; `unsupported` is a rule shape it will not guess at.
            if vjp_plan(name)[0] not in ("partner", "output", "quotient"):
                continue
            try:
                specs[f"{name}_vjp"] = vjp_spec(cores, name)
            except ValueError:
                continue
    return specs


#: Dekker's splitting constant for binary64: ``2**27 + 1``. Splitting a
#: 53-bit significand at the halfway point is what lets the four partial
#: products be formed without any of them rounding, which is the whole
#: mechanism -- the constant is structural, not tuned.
_SPLITTER = float(2 ** 27 + 1)

TWO_PRODUCT_SOURCE = """
def two_product(a, b, p, e, n):
    for i in range(n):
        av = a[i]
        bv = b[i]
        prod = av * bv
        ca = {splitter} * av
        ah = ca - (ca - av)
        al = av - ah
        cb = {splitter} * bv
        bh = cb - (cb - bv)
        bl = bv - bh
        p[i] = prod
        e[i] = ((ah * bh - prod) + ah * bl + al * bh) + al * bl
    return p
""".format(splitter=repr(_SPLITTER))


def two_product_oracle(a, b, p, e, n):
    """The exact product, split into a primal and its residual.

    An INDEPENDENT oracle, not a transcription of the kernel: it computes
    ``a * b`` over the rationals, where no rounding exists, and takes the
    residual by exact subtraction. Dekker's identity claims that residual
    is representable in one more double -- so admitting the kernel against
    this reference tests the CLAIM rather than confirming the kernel
    reproduces its own arithmetic.

    This is also what would make a fused variant checkable. ``fma(a, b,
    -prod)`` computes the same residual by a different route, and both
    being admitted against this same oracle is the evidence that they
    agree, rather than trusting that a capability flag implies it.
    """

    from fractions import Fraction

    for index in range(int(n)):
        left = float(a[index])
        right = float(b[index])
        product = left * right
        p[index] = product
        e[index] = float(
            Fraction(left) * Fraction(right) - Fraction(product)
        )
    return p


def two_product_spec():
    """The banked ``two_product`` kernel: Dekker's split, oracle-verified.

    The dual here is exactly the expression an optimizer deletes: ``ca -
    (ca - av)`` is algebraically ``av``, and the residual is algebraically
    zero. Nothing in the ingestion path folds arithmetic, so this compiles
    intact under a contract that licenses no inexact identity -- but under
    ``fast`` the toolchain's reassociation is free to evaporate all of it.
    That makes this kernel the concrete subject for the prove-versus-fast
    comparison: identical results mean the protection holds, and any
    difference at all means it does not.
    """

    from ...compiler.kernel_bank import KernelSpec

    return KernelSpec(
        name="two_product",
        source=TWO_PRODUCT_SOURCE,
        function_name="two_product",
        reference=two_product_oracle,
        parameter_order=("a", "b", "p", "e", "n"),
        size_parameters=("n",),
        # Spread over many binades so the residual is exercised where it is
        # large and where it is subnormal-adjacent. A narrow band would let
        # a kernel that silently drops the low limb still admit.
        example_inputs=lambda sizes, rng: {
            "a": rng.uniform(-1.0, 1.0, sizes["n"])
            * np.exp2(rng.integers(-40, 40, sizes["n"]).astype(np.float64)),
            "b": rng.uniform(-1.0, 1.0, sizes["n"])
            * np.exp2(rng.integers(-40, 40, sizes["n"]).astype(np.float64)),
            "p": np.zeros(sizes["n"]),
            "e": np.zeros(sizes["n"]),
            "n": int(sizes["n"]),
        },
        extents={"a": ("n",), "b": ("n",), "p": ("n",), "e": ("n",)},
    )


__all__ = [
    "TWO_PRODUCT_SOURCE",
    "two_product_oracle",
    "two_product_spec",
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


def variant_matrix(cores: tuple[str, ...] | None = None,
                   accuracies: tuple[int, ...] | None = None) -> dict:
    """Every core at every accuracy, as specs the bank can build.

    The matrix is the point: a core is not one program. Sized for eight digits
    it is a handful of terms, sized for thirty-two it is several times that,
    and which one a caller should use depends on what they will do with the
    answer. Enumerating them means the choice is made from measured cells
    rather than from one program and an assumption about the rest.
    """

    from . import signal_symbolic as _proof

    names = tuple(cores) if cores else tuple(
        name for name in _proof.TRANSCENDENTALS if name in _proof.CORE_RADII
    )
    widths = tuple(accuracies) if accuracies else ACCURACY_AXIS
    specs = {}
    for name in names:
        for digits in widths:
            try:
                spec = kernel_spec(None, name, digits=digits)
            except Exception:
                # A cell that cannot be built is left out rather than faked;
                # the caller sees which combinations exist.
                continue
            specs[spec.name] = spec
    return specs
