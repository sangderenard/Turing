"""Single-boundary execution of elementwise C-backend primitive programs.

This is a small native execution packet, not an autograd tape or a general
graph IR.  It exists to amortize Python/CFFI crossings and to provide a stable
prototype for passing precomposed primitive work to a native host such as
Nodus.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from typing import Any, Sequence

from .c_backend import C, CTensor, ffi


@dataclass(frozen=True)
class PrimitiveInstruction:
    """One equally-shaped elementwise primitive operation."""

    op: str
    out_slot: int
    left_slot: int
    right_slot: int | None = None
    right_scalar: float | None = None
    reverse: bool = False


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
class PrimitiveProgram:
    """A validated elementwise program executed through one CFFI call."""

    instructions: Sequence[PrimitiveInstruction]
    feed_count: int
    slot_count: int
    output_slot: int

    def _validate_feeds(self, feeds: Sequence[CTensor]):
        if len(feeds) != self.feed_count or not feeds:
            raise ValueError(f"expected {self.feed_count} non-empty feeds")
        shape = feeds[0].shape
        if any(feed.shape != shape for feed in feeds):
            raise ValueError("primitive-program feeds must have identical shapes")
        if self.slot_count < self.feed_count or not 0 <= self.output_slot < self.slot_count:
            raise ValueError("invalid primitive-program slot layout")
        return shape

    def _native_instructions(self):
        native = ffi.new("CTensorPrimitiveInstruction[]", len(self.instructions))
        for index, instruction in enumerate(self.instructions):
            try:
                native[index].op = getattr(C, _OP_NAMES[instruction.op])
            except KeyError as exc:
                raise ValueError(f"unsupported primitive op: {instruction.op}") from exc
            native[index].out_slot = instruction.out_slot
            native[index].left_slot = instruction.left_slot
            if instruction.right_slot is not None and instruction.right_scalar is not None:
                raise ValueError("an instruction cannot have both slot and scalar right operands")
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

    def prepare(self, feeds: Sequence[CTensor]) -> "PreparedPrimitiveProgram":
        """Bind reusable native slots for repeated execution."""
        shape = self._validate_feeds(feeds)
        slots = list(feeds) + [
            CTensor(shape) for _ in range(self.slot_count - self.feed_count)
        ]
        calculator = None
        calculator_mode = os.environ.get(
            "TENSOR_CALCULATOR_PROGRAMS", "auto"
        ).lower()
        use_calculator = (
            calculator_mode in ("1", "true", "yes")
            or (
                calculator_mode == "auto"
                and int(os.environ.get("TENSOR_CALCULATOR_WORKERS", "0")) > 0
            )
        )
        if use_calculator:
            try:
                from .native_calculator import get_native_calculator
                calculator = get_native_calculator()
            except Exception:
                calculator = None
        if calculator is not None:
            return calculator.prepare_program(self, slots)
        return PreparedPrimitiveProgram(
            self, slots, self._native_instructions(), prod(shape)
        )

    def execute(self, feeds: Sequence[CTensor]) -> CTensor:
        shape = self._validate_feeds(feeds)
        native = self._native_instructions()
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
            raise ValueError("native primitive-program validation failed")
        return output


class PreparedPrimitiveProgram:
    """A primitive program with stable native slots for repeated execution."""

    def __init__(self, program, slots, native_instructions, element_count):
        self.program = program
        self.slots = slots
        self.native_instructions = native_instructions
        self.element_count = element_count
        self.slot_ptrs = ffi.new(
            "double*[]", [slot.as_c_ptr() for slot in self.slots]
        )

    @property
    def output(self) -> CTensor:
        return self.slots[self.program.output_slot]

    def execute(self) -> CTensor:
        ok = C.ctensor_execute_primitive_program_slots(
            self.native_instructions,
            len(self.program.instructions),
            self.slot_ptrs,
            self.program.slot_count,
            self.element_count,
        )
        if not ok:
            raise ValueError("native primitive-program validation failed")
        return self.output


@dataclass(frozen=True)
class CapturedPrimitiveProgram:
    """A primitive program plus the ordered tensor roots captured as feeds."""

    program: PrimitiveProgram
    feeds: Sequence[Any]

    def c_feeds(self) -> list[CTensor]:
        """Materialize captured roots in the current double-only C ABI."""
        result = []
        for feed in self.feeds:
            values = feed.tolist()
            result.append(CTensor.from_list(values, tuple(feed.shape)))
        return result

    def execute_c(self) -> CTensor:
        return self.program.execute(self.c_feeds())


def compile_elementwise_tape(tape, output: Any) -> CapturedPrimitiveProgram:
    """Lower an output's eligible forward autograd trace to a primitive program.

    The compiler follows real ``GradTape`` parent ordering.  It accepts only
    equal-shape elementwise operations and scalar constants; reductions,
    broadcasting, indexing, and other regions remain explicit lowering
    boundaries.
    """

    aliases = {
        "div": "truediv",
        "radd": "add",
        "rsub": "sub",
        "rmul": "mul",
        "rtruediv": "truediv",
        "rpow": "pow",
        "less": "lt",
        "less_equal": "le",
        "greater": "gt",
        "greater_equal": "ge",
        "equal": "eq",
        "not_equal": "ne",
    }
    unary = {
        "sqrt", "exp", "log", "neg", "abs", "round", "trunc", "floor",
        "ceil", "isfinite", "isnan", "isinf", "logical_not",
    }
    binary = set(_OP_NAMES) - unary
    feed_slots: dict[int, int] = {}
    result_slots: dict[int, int] = {}
    feeds: list[Any] = []
    instructions: list[PrimitiveInstruction] = []

    def is_tensor(value):
        return hasattr(value, "data") and hasattr(value, "shape")

    def is_scalar(value):
        return isinstance(value, (int, float, bool)) or (
            is_tensor(value) and tuple(value.shape) in ((), (1,))
        )

    def scalar_value(value):
        return float(value.item() if is_tensor(value) else value)

    def feed_slot(value):
        identity = id(value)
        if identity not in feed_slots:
            feed_slots[identity] = len(feeds)
            feeds.append(value)
        return feed_slots[identity]

    def collect_feeds(value):
        node = tape._nodes.get(id(value))
        if node is None:
            if is_tensor(value) and not is_scalar(value):
                feed_slot(value)
            return
        for item in node.ctx["inputs"]:
            if is_tensor(item):
                collect_feeds(item)

    collect_feeds(output)

    def lower(value):
        identity = id(value)
        if identity in result_slots:
            return result_slots[identity]
        node = tape._nodes.get(identity)
        if node is None:
            if not is_tensor(value):
                raise ValueError("non-tensor trace root cannot become a feed")
            return feed_slot(value)

        original_op = node.op
        op = aliases.get(original_op, original_op)
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

        input_slots = {
            id(item): lower(item) for item in tensor_inputs
        }
        out_slot = len(feeds) + len(instructions)
        if op in unary and len(inputs) == 1:
            instruction = PrimitiveInstruction(
                op, out_slot, input_slots[id(inputs[0])]
            )
        elif op in binary and len(inputs) == 2:
            left, right = inputs
            if not is_scalar(left) and not is_scalar(right):
                instruction = PrimitiveInstruction(
                    op,
                    out_slot,
                    input_slots[id(left)],
                    right_slot=input_slots[id(right)],
                )
            elif not is_scalar(left) and is_scalar(right):
                instruction = PrimitiveInstruction(
                    op, out_slot, input_slots[id(left)],
                    right_scalar=scalar_value(right),
                    reverse=original_op.startswith("r"),
                )
            elif is_scalar(left) and not is_scalar(right):
                instruction = PrimitiveInstruction(
                    op, out_slot, input_slots[id(right)],
                    right_scalar=scalar_value(left), reverse=True,
                )
            else:
                raise ValueError(f"{op} has an unsupported operand layout")
        else:
            raise ValueError(f"{op} is not in the primitive elementwise region")
        instructions.append(instruction)
        result_slots[identity] = out_slot
        return out_slot

    output_slot = lower(output)
    program = PrimitiveProgram(
        instructions=tuple(instructions),
        feed_count=len(feeds),
        slot_count=len(feeds) + len(instructions),
        output_slot=output_slot,
    )
    return CapturedPrimitiveProgram(program, tuple(feeds))
