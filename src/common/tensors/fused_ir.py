"""Lightweight core of the established :mod:`abstract_nn.fused_program` IR.

The full builder and runner intentionally live in ``abstract_nn``.  Backend
lowerers import this module so using the C or GLSL backend does not initialize
the neural-network stack.  These are the same public IR classes re-exported by
``abstract_nn.fused_program``; this module is not a second program format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class Meta:
    """Per-id snapshot of tensor metadata."""

    shape: Iterable[int] | None = None
    dtype: str | None = None
    device: str | None = None


@dataclass
class OpStep:
    """Single linearised tensor operation."""

    step_id: int
    op_name: str
    input_ids: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)
    result_id: int = -1
    mode_sensitive: bool = False
    level: Optional[int] = None


@dataclass
class FusedProgram:
    """Unified program representation for AbstractTensor graphs."""

    version: int
    feeds: Set[int]
    steps: List[OpStep]
    outputs: Dict[str, int]
    state_in: Set[int] | None = None
    meta: Dict[int, Meta] | None = None
    extras: Dict[str, int] | None = None


ELEMENTWISE_ALIASES = {
    "div": "truediv",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
}

ELEMENTWISE_UNARY = frozenset(
    {
        "sqrt",
        "exp",
        "log",
        "neg",
        "abs",
        "round",
        "trunc",
        "floor",
        "ceil",
        "isfinite",
        "isnan",
        "isinf",
        "logical_not",
    }
)

ELEMENTWISE_BINARY = frozenset(
    {
        "add",
        "sub",
        "mul",
        "truediv",
        "pow",
        "mod",
        "floordiv",
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "maximum",
        "minimum",
    }
)


def canonical_elementwise_op(op: str) -> tuple[str, bool]:
    """Return the canonical AbstractTensor op name and operand reversal flag."""

    name = ELEMENTWISE_ALIASES.get(op, op)
    known = ELEMENTWISE_UNARY | ELEMENTWISE_BINARY
    if name in known:
        return name, False
    if name[:1] in ("i", "r"):
        base = ELEMENTWISE_ALIASES.get(name[1:], name[1:])
        if base in known:
            return base, name[0] == "r"
    raise KeyError(op)


def ordered_feed_ids(program: FusedProgram) -> tuple[int, ...]:
    """Return stable feed order used at backend boundaries."""

    explicit = getattr(program, "feed_order", None)
    if explicit is not None:
        return tuple(explicit)
    return tuple(sorted(program.feeds))


def primary_output_id(program: FusedProgram) -> int:
    """Return the sole output accepted by equal-shape fused backends."""

    if len(program.outputs) != 1:
        raise ValueError("elementwise fused backends require exactly one output")
    return next(iter(program.outputs.values()))
