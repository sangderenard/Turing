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


def _literal(value: float) -> str:
    """Round-trippable decimal for a coefficient baked into source."""

    return repr(float(value))


def _horner_expression(variable: str, coefficients: tuple[float, ...]) -> str:
    """Nested Horner as source text, innermost coefficient first."""

    expression = _literal(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        expression = f"({_literal(coefficient)} + {variable} * {expression})"
    return expression


def _require_structured(cores: _signal.CoreSet) -> tuple[Any, Any]:
    sine, cosine = cores["sin"], cores["cos"]
    if sine.family != "structured" or cosine.family != "structured":
        raise ValueError(
            "circular kernel source needs structured cores; got "
            f"{sine.family!r}/{cosine.family!r}"
        )
    return sine, cosine


def _reduction_lines(cores: _signal.CoreSet, phase: str) -> str:
    """The octant reduction and both polynomials, as loop-body source.

    Shared verbatim by the forward and the VJP so the two cannot drift in
    their reduction -- which is where the accuracy actually lives.

    Both cores are evaluated every iteration and the octant selects between
    them: each core serves the octant where the OTHER has its zero, so neither
    polynomial is ever asked to be accurate across its own root.
    """

    sine, cosine = _require_structured(cores)
    return (
        f"        v = x[i]{phase}\n"
        f"        t = v * {_literal(1.0 / math.tau)}\n"
        f"        s = t * 4.0 + 0.5\n"
        f"        k = s - (s % 1.0)\n"
        f"        r = (t - k * 0.25) * {_literal(math.tau)}\n"
        f"        w = k * 0.25\n"
        f"        q = k - (w - (w % 1.0)) * 4.0\n"
        f"        u = r * r\n"
        f"        sp = r * {_horner_expression('u', sine.values)}\n"
        f"        cp = {_horner_expression('u', cosine.values)}\n"
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


#: The backward rules this module can author: those whose derivative is a
#: signed multiple of a PARTNER function of the same argument.
_PARTNER_RULE = re.compile(
    r"^gx = unbroadcast\((-?)g \* (\w+)\(x\), x\.shape\)$"
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


def kernel_reference(source: str, name: str) -> Callable[..., Any]:
    """The oracle: the same source, executed as Python.

    Deliberately not a second implementation. A hand-written twin would be
    answering "do two authors agree", which is not the question bank
    admission asks.
    """

    namespace: dict[str, Any] = {}
    exec(compile(source, f"<signal-kernel {name}>", "exec"), namespace)
    return namespace[name]


def kernel_spec(cores: _signal.CoreSet, name: str):
    """One :class:`KernelSpec` for the forward of ``name``."""

    from ...compiler.kernel_bank import KernelSpec

    source = circular_kernel_source(cores, name)
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


def signal_kernel_specs(quality: str = "audio", *,
                        include_vjp: bool = True) -> Mapping[str, Any]:
    """Every kernel this module can author, at one baked quality."""

    cores = _signal.signal_math(quality).cores
    specs = {name: kernel_spec(cores, name) for name in ("sin", "cos")}
    if include_vjp:
        specs.update({
            f"{name}_vjp": vjp_spec(cores, name) for name in ("sin", "cos")
        })
    return specs


__all__ = [
    "circular_kernel_source",
    "circular_vjp_source",
    "kernel_reference",
    "kernel_spec",
    "signal_kernel_specs",
    "vjp_spec",
]
