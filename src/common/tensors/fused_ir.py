"""IR for resolving one bounded numeric segment.

``FusedProgram`` is **not the Turing compiler** and is not Turing's model of a
whole program.  It contains only the data dependencies needed to resolve a
flat numeric region after the compiler has already decided that the region is
eligible for numeric fusion.  It has no object identity, class organization,
control flow, process-graph ownership, map permissions, shell resources, or
deployment policy.

The full builder and runner intentionally live in ``abstract_nn``.  Numeric
backend lowerers import this lightweight module so using the C or GLSL backend
does not initialize the neural-network stack.  These are the same public IR
classes re-exported by ``abstract_nn.fused_program``; this module is not a
second program format.  A compiler-managed method or process may contain zero,
one, or many of these numeric segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


@dataclass
class Meta:
    """Per-id snapshot of tensor metadata.

    A value can either own storage (the default: ``source_id`` is ``None``)
    or be an explicit strided view over another value's buffer -- the same
    offset/stride composition ``nodus``'s ``TensorDesc``/``TensorStrides``
    use for its native tensor views. ``offset``/``stride`` are measured in
    elements of the source buffer, addressed per flat program index ``i`` as
    ``source[offset + i * stride]``. This is IR-level and backend-agnostic:
    any backend's per-index addressing consults it instead of assuming every
    operand is its own contiguous, stride-1 buffer.

    ``shape`` is the concrete extent observed during the one discovery trace
    that recorded this value -- correct for that trace, not necessarily for
    every real run.  ``shape_source_ids`` gives each dimension the same
    "usually absent, sometimes a real reference" treatment ``source_id``
    already gives storage: ``None`` at a position means that dimension is a
    genuine compile-time constant (the overwhelming common case, and the
    only case any backend consumes today); a ProcessGraph node id at a
    position instead means that dimension's real extent is whatever that
    node computes at actual runtime, and ``shape[i]`` is only the value one
    discovery trace happened to observe there.  A dimension whose origin
    cannot be traced at all falls back to ``None`` here the same way any
    other unresolvable value in this compiler becomes a plain ``Input`` --
    this field records a known origin, it does not invent one.
    """

    shape: Iterable[int] | None = None
    dtype: str | None = None
    device: str | None = None
    source_id: int | None = None
    offset: int = 0
    stride: int = 1
    shape_source_ids: tuple[int | None, ...] | None = None


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


def resolve_view_source(meta: Mapping[int, Meta] | None, value_id: int) -> int:
    """The buffer-owning value id backing ``value_id`` (itself, if none)."""

    if meta is None:
        return value_id
    entry = meta.get(value_id)
    if entry is None or entry.source_id is None:
        return value_id
    return resolve_view_source(meta, entry.source_id)


def view_offset_stride(meta: Mapping[int, Meta] | None, value_id: int) -> tuple[int, int]:
    """``(offset, stride)`` in elements for ``value_id`` against its buffer."""

    if meta is None:
        return 0, 1
    entry = meta.get(value_id)
    if entry is None:
        return 0, 1
    return entry.offset, entry.stride


# Trailing-axis reductions with a known, associative binary fold and
# identity element -- the set this IR can unroll into elementwise steps.
# Fold op names are the canonical ELEMENTWISE_BINARY vocabulary, since each
# fold becomes an ordinary binary OpStep between two lane views.
AXIS_REDUCTION_FOLDS: dict[str, tuple[str, float]] = {
    "sum": ("add", 0.0),
    "mean": ("add", 0.0),
    "prod": ("mul", 1.0),
    "min": ("minimum", float("inf")),
    "max": ("maximum", float("-inf")),
}


def unroll_feed_axis_reductions(program: "FusedProgram") -> "FusedProgram":
    """Replace a trailing-axis reduction over a dense feed buffer with an
    elementwise fold over ``K`` strided views of that same buffer.

    A flat ``run(count, ...)`` ABI has no memory contract for an externally
    supplied ``(..., K)`` feed -- nothing describes how its layout maps to a
    single byte offset -- so a reduction over a genuine feed buffer has
    always been an unlowerable shortfall for such a backend. The view
    descriptor on :class:`Meta` (``source_id``/``offset``/``stride``) now
    gives every K-th lane of that same buffer its own value id, so the
    reduction becomes an ordinary elementwise fold: no backend needs a
    dedicated reduction control path to run it.

    Only reductions whose operand is itself a feed (a value with no
    producing step) are eligible. A reduction over a value computed inside
    the program -- e.g. a broadcast composition of two lower-rank feeds --
    is left untouched; a backend that recomputes such a grid per fold step
    keeps doing so, unchanged by this pass.
    """

    meta: dict[int, Meta] = dict(program.meta or {})
    producers = {step.result_id for step in program.steps}
    used_ids = {step.result_id for step in program.steps} | set(program.feeds)
    next_id = (max(used_ids) + 1) if used_ids else 0

    def fresh_id() -> int:
        nonlocal next_id
        value_id = next_id
        next_id += 1
        return value_id

    rewritten: List[OpStep] = []
    changed = False
    new_feed_ids: set[int] = set()
    retired_feed_ids: set[int] = set()
    for step in program.steps:
        axis = step.attrs.get("axis")
        fold = AXIS_REDUCTION_FOLDS.get(step.op_name)
        if axis is None or fold is None or len(step.input_ids) != 1:
            rewritten.append(step)
            continue
        source_id = step.input_ids[0]
        if source_id in producers:
            rewritten.append(step)
            continue
        entry = meta.get(source_id)
        shape = tuple(entry.shape) if entry is not None and entry.shape is not None else None
        if shape is None or not shape:
            rewritten.append(step)
            continue
        norm_axis = axis if axis >= 0 else len(shape) + axis
        if norm_axis != len(shape) - 1:
            rewritten.append(step)
            continue
        lanes = int(shape[-1])
        if lanes <= 0:
            rewritten.append(step)
            continue
        changed = True
        retired_feed_ids.add(source_id)
        reduced_shape = shape[:-1]
        base_source = resolve_view_source(program.meta, source_id)
        base_offset, base_stride = view_offset_stride(program.meta, source_id)
        fold_op, _identity = fold
        final_id = fresh_id() if step.op_name == "mean" else step.result_id
        if lanes == 1:
            # A single lane needs no fold; the sole view already is the value.
            meta[final_id] = Meta(
                shape=reduced_shape, dtype=entry.dtype, device=entry.device,
                source_id=base_source, offset=base_offset, stride=base_stride,
            )
            new_feed_ids.add(final_id)
        else:
            view_ids = []
            for lane in range(lanes):
                view_id = fresh_id()
                meta[view_id] = Meta(
                    shape=reduced_shape, dtype=entry.dtype, device=entry.device,
                    source_id=base_source,
                    offset=base_offset + lane * base_stride,
                    stride=base_stride * lanes,
                )
                view_ids.append(view_id)
            new_feed_ids.update(view_ids)
            acc = view_ids[0]
            for index, lane_id in enumerate(view_ids[1:], start=1):
                is_last = index == lanes - 1
                rewritten.append(OpStep(
                    step_id=len(rewritten), op_name=fold_op,
                    input_ids=[acc, lane_id],
                    result_id=final_id if is_last else fresh_id(),
                ))
                acc = rewritten[-1].result_id
        if step.op_name == "mean":
            rewritten.append(OpStep(
                step_id=len(rewritten), op_name="truediv",
                input_ids=[final_id], attrs={"right_scalar": float(lanes)},
                result_id=step.result_id,
            ))

    if not changed:
        return program

    rewritten = [
        OpStep(
            step_id=index, op_name=step.op_name, input_ids=step.input_ids,
            attrs=step.attrs, result_id=step.result_id,
            mode_sensitive=step.mode_sensitive, level=step.level,
        )
        for index, step in enumerate(rewritten)
    ]
    feeds = (set(program.feeds) - retired_feed_ids) | new_feed_ids
    return FusedProgram(
        version=program.version, feeds=feeds, steps=rewritten,
        outputs=dict(program.outputs), state_in=program.state_in,
        meta=meta, extras=program.extras,
    )


@dataclass
class FusedProgram:
    """Dependency-resolved IR for one flat numeric segment.

    Despite the historical name, this object is neither a compiler nor a
    complete program.  ``feeds`` name values entering the numeric segment,
    ``steps`` are its linearly ordered tensor operations, and ``outputs`` name
    values leaving it.  The surrounding ProcessGraph, control IR, object/class
    map, permissions, method identity, shell, and runtime remain outside this
    object and must not be inferred from it.

    Backends consume a ``FusedProgram`` only after a larger compilation has
    selected such a numeric region.  Treating it as the compiler boundary
    discards the non-numeric structure that Turing is required to retain.
    """

    version: int
    feeds: Set[int]
    steps: List[OpStep]
    outputs: Dict[str, int]
    state_in: Set[int] | None = None
    meta: Dict[int, Meta] | None = None
    extras: Dict[str, int] | None = None


def flatten_tensor_constant(values: Any) -> tuple[float, ...]:
    """Return a tensor constructor payload in row-major scalar order.

    tensor_from_list is a backend-neutral creation operation. Keeping this
    small structural interpretation beside the IR prevents individual
    backends (or ProcessGraph adapters) from inventing incompatible ideas of
    what its nested values attribute means.
    """

    flattened: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if isinstance(value, Real):
            flattened.append(float(value))
            return
        raise TypeError(
            "tensor_from_list values must be nested numeric sequences; "
            f"found {type(value).__name__}"
        )

    visit(values)
    if not flattened:
        raise ValueError("tensor_from_list values must not be empty")
    return tuple(flattened)


def uniform_tensor_constant(values: Any) -> float | None:
    """Return the broadcast scalar represented by a uniform tensor value.

    Captured scalar literals are commonly materialized over the probe domain
    before they reach FusedProgram (for example [2, 2, 2, 2]). They remain
    semantically scalar broadcasts and may be emitted as immediates. A
    genuinely varying tensor returns None and must stay a tensor.
    """

    try:
        flattened = flatten_tensor_constant(values)
    except (TypeError, ValueError):
        return None
    first = flattened[0]
    return first if all(value == first for value in flattened[1:]) else None


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
        "sign",
        "round",
        "trunc",
        "floor",
        "ceil",
        "isfinite",
        "isnan",
        "isinf",
        "logical_not",
        "invert",
        "int_trunc",
        "zext",
        "sext",
        "fptosi",
        "fptoui",
        "sitofp",
        "uitofp",
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
        "bitand",
        "bitor",
        "bitxor",
        "shl",
        "shr",
        "logical_and",
        "logical_or",
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
    # ProcessGraph nodes built from a SymPy expression carry SSA-Handler-style
    # capitalized spellings ("Add", "Mul", "Pow", ...; see
    # symbolic_process_graph.py's _SYMPY_TO_CANONICAL) rather than this
    # module's lowercase tape-op vocabulary. Accept that spelling here, once,
    # instead of every caller lowercasing before calling in.
    lowered = name.lower()
    if lowered != name and lowered in known:
        return lowered, False
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
