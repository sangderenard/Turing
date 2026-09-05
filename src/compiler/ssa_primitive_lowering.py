"""Lower metadata-rich SSA into the established backend-neutral FusedProgram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    OpStep,
    canonical_elementwise_op,
)
from ..transmogrifier.ssa import Instr


# Span-memory initialisation constructors and their implicit fill scalar.
# ``None`` means the value must be supplied explicitly (``full``/``fill``).
_FILL_DEFAULTS: Dict[str, Optional[float]] = {
    "fill": None,
    "zeros": 0.0,
    "zeros_like": 0.0,
    "empty": 0.0,
    "empty_like": 0.0,
    "ones": 1.0,
    "ones_like": 1.0,
    "full": None,
    "full_like": None,
}



@dataclass(frozen=True)
class LoweringIssue:
    value_id: int
    op: str
    reason: str


@dataclass(frozen=True)
class FusedLoweringResult:
    program: Optional[FusedProgram]
    feed_value_ids: Tuple[int, ...]
    issues: Tuple[LoweringIssue, ...]
    lowered_value_ids: frozenset[int]

    @property
    def complete(self) -> bool:
        return self.program is not None and not self.issues

    def require_complete(self) -> FusedProgram:
        if not self.complete:
            details = "; ".join(
                f"%t{issue.value_id} {issue.op}: {issue.reason}"
                for issue in self.issues
            )
            raise ValueError(f"SSA FusedProgram lowering is incomplete: {details}")
        assert self.program is not None
        return self.program


def lower_ssa_to_fused_program(
    instrs: Sequence[Instr],
) -> FusedLoweringResult:
    """Lower equal-shape elementwise SSA to FusedProgram.

    ``nand`` and ``select`` are expressed as ordinary canonical FusedProgram
    steps. Shape-changing BitOps primitives remain structured boundaries until
    an appropriate non-elementwise backend region accepts them.
    """

    feed_ids = tuple(instr.res.id for instr in instrs if instr.op == "input")
    available = set(feed_ids)
    constants: Dict[int, float] = {}
    steps: list[OpStep] = []
    issues: list[LoweringIssue] = []
    outputs: dict[str, int] = {}
    next_value_id = max(
        (instr.res.id for instr in instrs),
        default=-1,
    ) + 1

    def fresh_id() -> int:
        nonlocal next_value_id
        result = next_value_id
        next_value_id += 1
        return result

    def operand(value_id: int):
        if value_id in available:
            return ("value", value_id)
        if value_id in constants:
            return ("scalar", constants[value_id])
        return None

    def append_step(
        op: str,
        result_id: int,
        input_ids: list[int],
        attrs: dict | None = None,
    ) -> None:
        steps.append(
            OpStep(
                step_id=len(steps),
                op_name=op,
                input_ids=input_ids,
                attrs=attrs or {},
                result_id=result_id,
            )
        )
        available.add(result_id)

    def binary_step(op: str, result_id: int, arg_ids: Sequence[int]) -> bool:
        if len(arg_ids) != 2:
            issues.append(LoweringIssue(result_id, op, "expected two operands"))
            return False
        left = operand(arg_ids[0])
        right = operand(arg_ids[1])
        if left is None or right is None:
            issues.append(LoweringIssue(result_id, op, "operand has no lowered value"))
            return False
        if left[0] == right[0] == "scalar":
            issues.append(
                LoweringIssue(result_id, op, "constant folding is not implemented")
            )
            return False
        if left[0] == right[0] == "value":
            append_step(op, result_id, [left[1], right[1]])
        elif left[0] == "value":
            append_step(op, result_id, [left[1]], {"right_scalar": right[1]})
        else:
            append_step(
                op,
                result_id,
                [right[1]],
                {"right_scalar": left[1], "reverse": True},
            )
        return True

    for instr in instrs:
        try:
            op, reverse = canonical_elementwise_op(instr.op)
        except KeyError:
            op, reverse = instr.op, False
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
                    LoweringIssue(result_id, op, "scalar-only output is unsupported")
                )
            else:
                outputs["result"] = operand(arg_ids[0])[1]
                available.add(result_id)
            continue

        if op == "random_source":
            append_step(op, result_id, [], dict(instr.attributes))
            continue

        op_lower = op.lower() if isinstance(op, str) else op
        if op_lower in _FILL_DEFAULTS:
            default_value = _FILL_DEFAULTS[op_lower]
            fill_value = instr.attributes.get(
                "fill_value", instr.attributes.get("value", default_value)
            )
            if fill_value is None:
                issues.append(
                    LoweringIssue(result_id, op, "fill requires an explicit fill_value")
                )
                continue
            shape = (
                tuple(instr.res.shape)
                if getattr(instr.res, "shape", ())
                else tuple(instr.attributes.get("shape", ()))
            )
            # Span-memory initialisation. Zero-fill is the calloc case and keeps
            # the ``zeros`` spelling; any other constant uses ``full``.
            append_step(
                "zeros" if float(fill_value) == 0.0 else "full",
                result_id,
                [],
                {"shape": shape, "fill_value": float(fill_value)},
            )
            continue

        if op in ELEMENTWISE_UNARY:
            if len(arg_ids) != 1:
                issues.append(LoweringIssue(result_id, op, "expected one operand"))
                continue
            arg = operand(arg_ids[0])
            if arg is None or arg[0] != "value":
                issues.append(
                    LoweringIssue(result_id, op, "unary operand must be a tensor value")
                )
                continue
            append_step(op, result_id, [arg[1]], {"reverse": reverse} if reverse else None)
            continue

        if op in ELEMENTWISE_BINARY:
            if binary_step(op, result_id, arg_ids) and reverse:
                steps[-1].attrs["reverse"] = not steps[-1].attrs.get("reverse", False)
            continue

        if op == "nand":
            product_id = fresh_id()
            if binary_step("mul", product_id, arg_ids):
                append_step("logical_not", result_id, [product_id])
            continue

        if op == "select":
            if len(arg_ids) != 3:
                issues.append(LoweringIssue(result_id, op, "expected condition/true/false"))
                continue
            values = [operand(value_id) for value_id in arg_ids]
            if any(item is None or item[0] != "value" for item in values):
                issues.append(
                    LoweringIssue(result_id, op, "select requires three tensor values")
                )
                continue
            condition, if_true, if_false = (item[1] for item in values)
            delta = fresh_id()
            weighted = fresh_id()
            append_step("sub", delta, [if_true, if_false])
            append_step("mul", weighted, [condition, delta])
            append_step("add", result_id, [if_false, weighted])
            continue

        issues.append(
            LoweringIssue(
                result_id,
                op,
                "not in the equal-shape FusedProgram vocabulary",
            )
        )

    if not outputs:
        for instr in reversed(instrs):
            if instr.res.id in available:
                outputs["result"] = instr.res.id
                break
    program = (
        FusedProgram(
            version=1,
            feeds=set(feed_ids),
            steps=steps,
            outputs=outputs,
        )
        if outputs and not issues
        else None
    )
    return FusedLoweringResult(
        program, feed_ids, tuple(issues), frozenset(available)
    )


# Transitional spelling for callers outside this repository. It returns a
# FusedProgram and must not be used to infer a second PrimitiveProgram IR.
lower_ssa_to_primitive_program = lower_ssa_to_fused_program
