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
a pack.

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


def circular_kernel_source(cores: _signal.CoreSet, name: str = "sin") -> str:
    """Elementwise ``sin``/``cos`` over radians, octant-reduced.

    Both cores are evaluated every iteration and the octant selects between
    them. That is the same shape the AbstractTensor surface uses, for the same
    reason: each core serves the octant where the other one has its zero, so
    neither polynomial is ever asked to be accurate across its own root.
    """

    if name not in ("sin", "cos"):
        raise ValueError(f"circular kernel must be sin or cos, got {name!r}")
    sine, cosine = cores["sin"], cores["cos"]
    if sine.family != "structured" or cosine.family != "structured":
        raise ValueError(
            "circular kernel source needs structured cores; got "
            f"{sine.family!r}/{cosine.family!r}"
        )
    # cos(v) is sin(v + tau/4); the shift is exact and keeps one code path.
    phase = "" if name == "sin" else f" + {_literal(math.tau / 4.0)}"
    return f'''
def {name}(x, y, n):
    for i in range(n):
        v = x[i]{phase}
        t = v * {_literal(1.0 / math.tau)}
        s = t * 4.0 + 0.5
        k = s - (s % 1.0)
        r = (t - k * 0.25) * {_literal(math.tau)}
        d = k * 0.25
        q = k - (d - (d % 1.0)) * 4.0
        u = r * r
        sp = r * {_horner_expression("u", sine.values)}
        cp = {_horner_expression("u", cosine.values)}
        if q < 1.0:
            y[i] = sp
        elif q < 2.0:
            y[i] = cp
        elif q < 3.0:
            y[i] = -sp
        else:
            y[i] = -cp
    return y
'''


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
    """One :class:`KernelSpec` for the bank, built from the baked cores."""

    from ...compiler.kernel_bank import KernelSpec

    source = circular_kernel_source(cores, name)
    return KernelSpec(
        name=name,
        source=source,
        function_name=name,
        reference=kernel_reference(source, name),
        parameter_order=("x", "y", "n"),
        size_parameters=("n",),
        example_inputs=lambda sizes, rng: {
            "x": rng.uniform(-8.0, 8.0, sizes["n"]),
            "y": np.zeros(sizes["n"]),
            "n": int(sizes["n"]),
        },
        extents={"x": ("n",), "y": ("n",)},
    )


def signal_kernel_specs(quality: str = "audio") -> Mapping[str, Any]:
    """Every kernel this module can author, at one baked quality."""

    cores = _signal.signal_math(quality).cores
    return {name: kernel_spec(cores, name) for name in ("sin", "cos")}


__all__ = [
    "circular_kernel_source",
    "kernel_reference",
    "kernel_spec",
    "signal_kernel_specs",
]
