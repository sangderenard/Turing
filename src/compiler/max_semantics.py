"""One rule per authored meaning of ``max``/``min``, shared by every backend.

The planner stamps ``max_semantics`` on a ``max``/``min`` instruction from
the call form that wrote it (``receiver_resolution_prepasses``):

``select``      builtin ``max(a, b, ...)``
``scan``        builtin ``max(xs)``
``reduce``      ``xs.max()`` with no axis
``reduce_axis`` ``xs.max(axis=k)``
``elementwise`` ``x.maximum(y)``
``emax``        ``x.emax(y)`` -- builtin ``max`` per element

Two step rules cover them. ``PYTHON`` is CPython's builtin: the running
value is replaced only when the next one is strictly greater, so a NaN
never wins a comparison and a leading NaN stays; a tie keeps the earlier
value. ``NUMPY`` is ``np.maximum`` and the ``np.max`` fold: a NaN on either
side is the result, otherwise the greater, and a tie takes the later value.
An unstamped ``Max``/``Min`` is compiler-introduced (a clamp, a bound) and
keeps the backend's own spelling.
"""

from __future__ import annotations

from typing import Any

PYTHON = "python"
NUMPY = "numpy"

_RULES = {
    "select": PYTHON,
    "scan": PYTHON,
    "reduce": NUMPY,
    "reduce_axis": NUMPY,
    "elementwise": NUMPY,
    "emax": PYTHON,
}


def max_step_rule(instruction: Any) -> str | None:
    """``PYTHON``, ``NUMPY``, or ``None`` for an unstamped instruction."""

    attributes = getattr(instruction, "attributes", None) or {}
    return _RULES.get(str(attributes.get("max_semantics") or ""))


def is_max(operation: str) -> bool | None:
    """True for a ``max`` spelling, False for ``min``, None otherwise."""

    name = str(operation).casefold()
    if name in {"max", "maximum", "amax", "emax"}:
        return True
    if name in {"min", "minimum", "amin", "emin"}:
        return False
    return None


def c_step(rule: str, greatest: bool, running: str, value: str) -> str:
    """The C expression of one step from ``running`` with ``value``."""

    better = ">" if greatest else "<"
    if rule == PYTHON:
        return f"((({value}) {better} ({running})) ? ({value}) : ({running}))"
    return (
        f"((({running}) != ({running})) ? ({running}) : "
        f"(({value}) != ({value})) ? ({value}) : "
        f"((({running}) {better} ({value})) ? ({running}) : ({value})))"
    )


def fortran_step(rule: str, greatest: bool, running: str, value: str) -> str:
    """The Fortran expression of one step (elementwise over arrays too).

    ``x /= x`` is the NaN test; gfortran's ``max``/``min`` intrinsics are
    not used because their NaN result is processor-dependent.
    """

    better = ">" if greatest else "<"
    if rule == PYTHON:
        return f"merge({value}, {running}, {value} {better} {running})"
    return (
        f"merge({running}, merge({value}, merge({running}, {value}, "
        f"{running} {better} {value}), {value} /= {value}), "
        f"{running} /= {running})"
    )


def fortran_reduction(rule: str, greatest: bool, array: str) -> str:
    """A whole-array ``max``/``min`` of a rank-1 ``array`` under ``rule``.

    ``maxval`` skips NaN. numpy's fold is NaN when any element is, and
    CPython's scan is NaN exactly when the first element is (a later NaN
    never wins a comparison, so the result is the extreme of the rest).
    """

    intrinsic = "maxval" if greatest else "minval"
    if rule == PYTHON:
        first = f"{array}(lbound({array}, 1))"
        return f"merge({first}, {intrinsic}({array}), {first} /= {first})"
    return (
        f"merge(sum({array}, mask={array} /= {array}), {intrinsic}({array}), "
        f"any({array} /= {array}))"
    )


__all__ = [
    "NUMPY", "PYTHON", "c_step", "fortran_reduction", "fortran_step",
    "is_max", "max_step_rule",
]
