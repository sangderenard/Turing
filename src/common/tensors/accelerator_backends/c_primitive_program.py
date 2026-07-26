"""One-boundary C execution of the established :class:`FusedProgram` IR.

``FusedProgram`` remains the semantic program.  This module only compiles its
equal-shape elementwise regions into a private slot plan matching the native C
ABI, then executes that plan through one CFFI call.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

from ..fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    ordered_feed_ids,
    primary_output_id,
)
from .c_backend import C, CTensor, ffi


_OP_NAMES = {
    "add": "CT_OP_ADD",
    "sub": "CT_OP_SUB",
    "mul": "CT_OP_MUL",
    "truediv": "CT_OP_DIV",
    "pow": "CT_OP_POW",
    "mod": "CT_OP_MOD",
    "floordiv": "CT_OP_FLOORDIV",
    "sqrt": "CT_OP_SQRT",
    "exp": "CT_OP_EXP",
    "log": "CT_OP_LOG",
    "neg": "CT_OP_NEG",
    "abs": "CT_OP_ABS",
    "round": "CT_OP_ROUND",
    "trunc": "CT_OP_TRUNC",
    "floor": "CT_OP_FLOOR",
    "ceil": "CT_OP_CEIL",
    "isfinite": "CT_OP_ISFINITE",
    "isnan": "CT_OP_ISNAN",
    "isinf": "CT_OP_ISINF",
    "logical_not": "CT_OP_LOGICAL_NOT",
    "lt": "CT_OP_LT",
    "le": "CT_OP_LE",
    "gt": "CT_OP_GT",
    "ge": "CT_OP_GE",
    "eq": "CT_OP_EQ",
    "ne": "CT_OP_NE",
    "maximum": "CT_OP_MAXIMUM",
    "minimum": "CT_OP_MINIMUM",
}


@dataclass(frozen=True)
class _SlotInstruction:
    op: str
    out_slot: int
    left_slot: int
    right_slot: int | None = None
    right_scalar: float | None = None
    reverse: bool = False


@dataclass(frozen=True)
class _SlotPlan:
    """Private C ABI layout compiled from a FusedProgram."""

    instructions: tuple[_SlotInstruction, ...]
    feed_ids: tuple[int, ...]
    slot_count: int
    output_slot: int

    @property
    def feed_count(self) -> int:
        return len(self.feed_ids)

    def validate_feeds(self, feeds: Sequence[CTensor]):
        if len(feeds) != self.feed_count or not feeds:
            raise ValueError(f"expected {self.feed_count} non-empty feeds")
        shape = feeds[0].shape
        if any(feed.shape != shape for feed in feeds):
            raise ValueError("fused-program feeds must have identical shapes")
        return shape

    def native_instructions(self):
        native = ffi.new("CTensorPrimitiveInstruction[]", len(self.instructions))
        for index, instruction in enumerate(self.instructions):
            native[index].op = getattr(C, _OP_NAMES[instruction.op])
            native[index].out_slot = instruction.out_slot
            native[index].left_slot = instruction.left_slot
            if instruction.right_slot is not None:
                native[index].right_kind = C.CT_OPERAND_SLOT
                native[index].right_slot = instruction.right_slot
            elif instruction.right_scalar is not None:
                native[index].right_kind = C.CT_OPERAND_SCALAR
                native[index].right_scalar = instruction.right_scalar
            else:
                native[index].right_kind = C.CT_OPERAND_NONE
            native[index].reverse = int(instruction.reverse)
        return native

    def prepare(self, feeds: Sequence[CTensor]) -> "PreparedFusedProgram":
        shape = self.validate_feeds(feeds)
        slots = list(feeds) + [
            CTensor(shape) for _ in range(self.slot_count - self.feed_count)
        ]
        return PreparedFusedProgram(
            self, slots, self.native_instructions(), prod(shape)
        )

    def execute(self, feeds: Sequence[CTensor]) -> CTensor:
        shape = self.validate_feeds(feeds)
        native = self.native_instructions()
        element_count = prod(shape)
        feed_ptrs = ffi.new("const double*[]", [feed.as_c_ptr() for feed in feeds])
        workspace = ffi.new("double[]", self.slot_count * element_count)
        output = CTensor(shape)
        ok = C.ctensor_execute_primitive_program(
            native,
            len(self.instructions),
            feed_ptrs,
            self.feed_count,
            workspace,
            self.slot_count,
            element_count,
            self.output_slot,
            output.as_c_ptr(),
        )
        if not ok:
            raise ValueError("native fused-program validation failed")
        return output


class PreparedFusedProgram:
    """A compiled FusedProgram with stable native slots."""

    def __init__(self, plan, slots, native_instructions, element_count):
        self.plan = plan
        # Compatibility for the native calculator's prepared wrapper.
        self.program = plan
        self.slots = slots
        self.native_instructions = native_instructions
        self.element_count = element_count
        self.slot_ptrs = ffi.new(
            "double*[]", [slot.as_c_ptr() for slot in self.slots]
        )

    @property
    def output(self) -> CTensor:
        return self.slots[self.plan.output_slot]

    def execute(self) -> CTensor:
        ok = C.ctensor_execute_primitive_program_slots(
            self.native_instructions,
            len(self.plan.instructions),
            self.slot_ptrs,
            self.plan.slot_count,
            self.element_count,
        )
        if not ok:
            raise ValueError("native fused-program validation failed")
        return self.output


def compile_fused_program(program: FusedProgram) -> _SlotPlan:
    """Compile an equal-shape FusedProgram into the private native slot ABI."""

    feed_ids = ordered_feed_ids(program)
    slots = {value_id: index for index, value_id in enumerate(feed_ids)}
    instructions: list[_SlotInstruction] = []

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
        if not step.input_ids or step.input_ids[0] not in slots:
            raise ValueError(
                f"FusedProgram step {step.step_id} reads an unavailable input"
            )
        left_slot = slots[step.input_ids[0]]
        right_slot = None
        if op in ELEMENTWISE_UNARY:
            if len(step.input_ids) != 1 or scalar is not None:
                raise ValueError(f"unary op {op} has an invalid operand layout")
        elif op in ELEMENTWISE_BINARY:
            if len(step.input_ids) == 2 and scalar is None:
                try:
                    right_slot = slots[step.input_ids[1]]
                except KeyError as exc:
                    raise ValueError(
                        f"FusedProgram step {step.step_id} reads an unavailable input"
                    ) from exc
            elif len(step.input_ids) != 1 or scalar is None:
                raise ValueError(f"binary op {op} has an invalid operand layout")
            scalar = None if scalar is None else float(scalar)
        out_slot = len(feed_ids) + len(instructions)
        slots[step.result_id] = out_slot
        instructions.append(
            _SlotInstruction(
                op,
                out_slot,
                left_slot,
                right_slot=right_slot,
                right_scalar=scalar,
                reverse=reverse,
            )
        )

    output_id = primary_output_id(program)
    if output_id not in slots:
        raise ValueError("FusedProgram output is not produced")
    return _SlotPlan(
        tuple(instructions), feed_ids, len(feed_ids) + len(instructions), slots[output_id]
    )


def _ordered_c_feeds(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> list[CTensor]:
    ids = ordered_feed_ids(program)
    if isinstance(feeds, Mapping):
        missing = set(ids) - set(feeds)
        if missing:
            raise ValueError(f"missing FusedProgram feeds: {sorted(missing)}")
        return [feeds[value_id] for value_id in ids]
    return list(feeds)


def execute_fused_program(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> CTensor:
    """Execute a FusedProgram through one CFFI call."""

    return compile_fused_program(program).execute(_ordered_c_feeds(program, feeds))


def prepare_fused_program(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> PreparedFusedProgram:
    """Prepare a FusedProgram for repeated one-boundary execution."""

    return compile_fused_program(program).prepare(_ordered_c_feeds(program, feeds))


@dataclass(frozen=True)
class CapturedFusedProgram:
    """A FusedProgram plus the tensor roots captured as its feed bindings."""

    program: FusedProgram
    feeds: Mapping[int, Any]

    def c_feeds(self) -> dict[int, CTensor]:
        return {
            feed_id: CTensor.from_list(value.tolist(), tuple(value.shape))
            for feed_id, value in self.feeds.items()
        }

    def execute_c(self) -> CTensor:
        return execute_fused_program(self.program, self.c_feeds())


def compile_elementwise_tape(tape, output: Any) -> CapturedFusedProgram:
    """Capture an eligible forward GradTape region as a FusedProgram.

    Reductions, broadcasting, indexing, and shape-changing operations are
    explicit boundaries.  No C-specific program representation escapes this
    lowering function.
    """

    feeds: dict[int, Any] = {}
    steps: list[OpStep] = []
    lowered: set[int] = set()
    metadata: dict[int, Meta] = {}

    def is_tensor(value):
        return hasattr(value, "data") and hasattr(value, "shape")

    def is_scalar(value):
        return isinstance(value, (int, float, bool)) or (
            is_tensor(value) and tuple(value.shape) in ((), (1,))
        )

    def scalar_value(value):
        return float(value.item() if is_tensor(value) else value)

    def lower(value):
        identity = id(value)
        if identity in lowered:
            return identity
        node = tape._nodes.get(identity)
        if node is None:
            if not is_tensor(value) or is_scalar(value):
                raise ValueError("non-tensor trace root cannot become a feed")
            feeds[identity] = value
            metadata[identity] = Meta(shape=tuple(value.shape))
            lowered.add(identity)
            return identity

        original_op = node.op
        try:
            op, prefix_reverse = canonical_elementwise_op(original_op)
        except KeyError as exc:
            raise ValueError(
                f"{original_op} is not in the elementwise FusedProgram region"
            ) from exc
        inputs = node.ctx["inputs"]
        tensor_inputs = [
            item for item in inputs if is_tensor(item) and not is_scalar(item)
        ]
        if not tensor_inputs:
            raise ValueError(f"{op} has no tensor input")
        expected_shape = tuple(tensor_inputs[0].shape)
        if tuple(value.shape) != expected_shape or any(
            tuple(item.shape) != expected_shape for item in tensor_inputs
        ):
            raise ValueError(f"{op} crosses an elementwise shape boundary")

        attrs: dict[str, Any] = {}
        input_ids: list[int]
        if op in ELEMENTWISE_UNARY and len(inputs) == 1:
            input_ids = [lower(inputs[0])]
        elif op in ELEMENTWISE_BINARY and len(inputs) == 2:
            left, right = inputs
            if not is_scalar(left) and not is_scalar(right):
                input_ids = [lower(left), lower(right)]
            elif not is_scalar(left) and is_scalar(right):
                input_ids = [lower(left)]
                attrs["right_scalar"] = scalar_value(right)
            elif is_scalar(left) and not is_scalar(right):
                input_ids = [lower(right)]
                attrs["right_scalar"] = scalar_value(left)
                attrs["reverse"] = True
            else:
                raise ValueError(f"{op} has an unsupported operand layout")
        else:
            raise ValueError(f"{op} has an unsupported operand layout")
        steps.append(
            OpStep(
                step_id=len(steps),
                op_name=op,
                input_ids=input_ids,
                attrs=attrs,
                result_id=identity,
            )
        )
        metadata[identity] = Meta(shape=tuple(value.shape))
        lowered.add(identity)
        return identity

    output_id = lower(output)
    program = FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=steps,
        outputs={"result": output_id},
        meta=metadata,
    )
    return CapturedFusedProgram(program, feeds)
