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
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "eq": "equal",
    "ne": "not_equal",
}

ELEMENTWISE_UNARY = frozenset(
    {
        "sqrt",
        "exp",
        "log",
        "tanh",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
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
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
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


def serialize_elementwise_fused_program(program: FusedProgram) -> str:
    """Serialize an equal-shape region for cross-language calculator replay.

    This is a textual transport for :class:`FusedProgram`, not another
    semantic program representation.  Operation names remain canonical and
    consumers perform their own value-id-to-storage lowering.
    """

    lines = [f"fused_program {program.version}"]
    for feed_id in ordered_feed_ids(program):
        lines.append(f"feed {feed_id}")

    for step in program.steps:
        try:
            op, prefix_reverse = canonical_elementwise_op(step.op_name)
        except KeyError as exc:
            raise ValueError(
                f"{step.op_name} is not in the elementwise FusedProgram region"
            ) from exc
        attrs = dict(step.attrs)
        reverse = bool(attrs.pop("reverse", False)) ^ prefix_reverse
        scalar = attrs.pop("right_scalar", None)
        if attrs:
            raise ValueError(
                f"FusedProgram step {step.step_id} has unsupported attrs: "
                f"{', '.join(sorted(attrs))}"
            )
        if op in ELEMENTWISE_UNARY:
            if len(step.input_ids) != 1 or scalar is not None:
                raise ValueError(f"unary op {op} has an invalid operand layout")
        elif len(step.input_ids) == 2 and scalar is None:
            pass
        elif len(step.input_ids) == 1 and scalar is not None:
            scalar = float(scalar)
        else:
            raise ValueError(f"binary op {op} has an invalid operand layout")

        tokens = [
            "step",
            str(step.step_id),
            op,
            str(step.result_id),
            str(len(step.input_ids)),
            *(str(value_id) for value_id in step.input_ids),
            "1" if scalar is not None else "0",
            format(scalar if scalar is not None else 0.0, ".17g"),
            "1" if reverse else "0",
        ]
        lines.append(" ".join(tokens))

    lines.append(f"output {primary_output_id(program)}")
    lines.append("end")
    return "\n".join(lines) + "\n"
