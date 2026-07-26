"""Lower metadata-rich SSA into the shared C/GLSL PrimitiveProgram shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..common.tensors.accelerator_backends.c_primitive_program import (
    PrimitiveInstruction,
    PrimitiveProgram,
)
from ..transmogrifier.ssa import Instr


_ALIASES = {
    "div": "truediv",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
}

_UNARY = {
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

_BINARY = {
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


@dataclass(frozen=True)
class LoweringIssue:
    value_id: int
    op: str
    reason: str


@dataclass(frozen=True)
class PrimitiveLoweringResult:
    program: Optional[PrimitiveProgram]
    feed_value_ids: Tuple[int, ...]
    issues: Tuple[LoweringIssue, ...]
    value_slots: Mapping[int, int]

    @property
    def complete(self) -> bool:
        return self.program is not None and not self.issues

    def require_complete(self) -> PrimitiveProgram:
        if not self.complete:
            details = "; ".join(
                f"%t{issue.value_id} {issue.op}: {issue.reason}"
                for issue in self.issues
            )
            raise ValueError(f"SSA primitive lowering is incomplete: {details}")
        return self.program


def lower_ssa_to_primitive_program(
    instrs: Sequence[Instr],
) -> PrimitiveLoweringResult:
    """Lower equal-shape elementwise SSA to the one-call backend program.

    `nand` and `select` are expanded into canonical primitive instructions.
    Shape-changing BitOps primitives are rejected as structured issues because
    the current PrimitiveProgram contract intentionally has equal-shaped slots.
    """

    feed_ids = tuple(instr.res.id for instr in instrs if instr.op == "input")
    slots: Dict[int, int] = {
        value_id: slot for slot, value_id in enumerate(feed_ids)
    }
    constants: Dict[int, float] = {}
    native: List[PrimitiveInstruction] = []
    issues: List[LoweringIssue] = []
    next_slot = len(feed_ids)
    output_slot: Optional[int] = None

    def allocate() -> int:
        nonlocal next_slot
        slot = next_slot
        next_slot += 1
        return slot

    def operand(value_id: int):
        if value_id in slots:
            return ("slot", slots[value_id])
        if value_id in constants:
            return ("scalar", constants[value_id])
        return None

    def binary_instruction(op: str, result_id: int, arg_ids: Sequence[int]) -> bool:
        if len(arg_ids) != 2:
            issues.append(LoweringIssue(result_id, op, "expected two operands"))
            return False
        left = operand(arg_ids[0])
        right = operand(arg_ids[1])
        if left is None or right is None:
            issues.append(LoweringIssue(result_id, op, "operand has no lowered value"))
            return False
        if left[0] == right[0] == "scalar":
            issues.append(LoweringIssue(result_id, op, "constant folding is not implemented"))
            return False
        out = allocate()
        if left[0] == "slot" and right[0] == "slot":
            native.append(
                PrimitiveInstruction(op, out, left[1], right_slot=right[1])
            )
        elif left[0] == "slot":
            native.append(
                PrimitiveInstruction(op, out, left[1], right_scalar=right[1])
            )
        else:
            native.append(
                PrimitiveInstruction(
                    op, out, right[1], right_scalar=left[1], reverse=True
                )
            )
        slots[result_id] = out
        return True

    for instr in instrs:
        op = _ALIASES.get(instr.op, instr.op)
        result_id = instr.res.id
        arg_ids = [arg.id for arg in instr.args]

        if op == "input":
            continue
        if op == "const":
            value = instr.attributes.get("value")
            if isinstance(value, (bool, int, float)):
                constants[result_id] = float(value)
            else:
                issues.append(
                    LoweringIssue(result_id, op, "constant is not a numeric scalar")
                )
            continue
        if op == "return":
            if len(arg_ids) != 1 or operand(arg_ids[0]) is None:
                issues.append(
                    LoweringIssue(result_id, op, "return value is not lowered")
                )
            elif operand(arg_ids[0])[0] == "scalar":
                issues.append(
                    LoweringIssue(result_id, op, "scalar-only output has no tensor slot")
                )
            else:
                output_slot = operand(arg_ids[0])[1]
                slots[result_id] = output_slot
            continue

        if op in _UNARY:
            if len(arg_ids) != 1:
                issues.append(LoweringIssue(result_id, op, "expected one operand"))
                continue
            arg = operand(arg_ids[0])
            if arg is None or arg[0] != "slot":
                issues.append(
                    LoweringIssue(result_id, op, "unary operand must be a tensor slot")
                )
                continue
            out = allocate()
            native.append(PrimitiveInstruction(op, out, arg[1]))
            slots[result_id] = out
            continue

        if op in _BINARY:
            binary_instruction(op, result_id, arg_ids)
            continue

        if op == "nand":
            if not binary_instruction("mul", result_id, arg_ids):
                continue
            product_slot = slots.pop(result_id)
            out = allocate()
            native.append(PrimitiveInstruction("logical_not", out, product_slot))
            slots[result_id] = out
            continue

        if op == "select":
            if len(arg_ids) != 3:
                issues.append(LoweringIssue(result_id, op, "expected condition/true/false"))
                continue
            condition, if_true, if_false = (operand(value_id) for value_id in arg_ids)
            if (
                condition is None
                or if_true is None
                or if_false is None
                or any(item[0] != "slot" for item in (condition, if_true, if_false))
            ):
                issues.append(
                    LoweringIssue(result_id, op, "select requires three tensor slots")
                )
                continue
            delta = allocate()
            weighted = allocate()
            out = allocate()
            native.extend(
                (
                    PrimitiveInstruction(
                        "sub", delta, if_true[1], right_slot=if_false[1]
                    ),
                    PrimitiveInstruction(
                        "mul", weighted, condition[1], right_slot=delta
                    ),
                    PrimitiveInstruction(
                        "add", out, if_false[1], right_slot=weighted
                    ),
                )
            )
            slots[result_id] = out
            continue

        issues.append(
            LoweringIssue(
                result_id,
                op,
                "not in the equal-shape primitive vocabulary",
            )
        )

    if output_slot is None:
        for instr in reversed(instrs):
            if instr.res.id in slots:
                output_slot = slots[instr.res.id]
                break
    program = (
        PrimitiveProgram(
            tuple(native),
            feed_count=len(feed_ids),
            slot_count=next_slot,
            output_slot=output_slot,
        )
        if output_slot is not None and not issues
        else None
    )
    return PrimitiveLoweringResult(program, feed_ids, tuple(issues), dict(slots))

