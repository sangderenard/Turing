"""Automatic per-region compute-shader eligibility for deployment lanes.

The standard for a finished deployment is that the compiler DECIDES where a
compute shader is valuable, rather than a product hand-picking one.  The
decision seam is the outlined deployment lane (``deployment_outlining``):
one callable body per independent iteration is exactly the shape a compute
dispatch wants (one invocation = one iteration), so eligibility is judged
there, against the real GLSL compute dialect
(``ssa_glsl_compute_backend``), and recorded as receipts:

- a lane whose body is straight-line and entirely within the GPU dialect is
  ELIGIBLE: the pooled CPU deploy and the GPU dispatch are then competing
  lowerings of the same proven region, and a backend/product may pick by
  cost;
- anything else is a NAMED refusal (internal calls, control flow, sequence
  effects, uncovered operations), so the next widening of the GPU dialect
  is an explicit list instead of a guess.

Nothing here fakes a dispatch: eligibility is a verdict, artifact emission
stays with the GLSL backend, and a refused lane keeps its pooled CPU
deploy.  Selection is deliberately conservative -- a wrong "eligible" would
put physics on a lane that cannot spell it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The operations the desktop-GLSL compute dialect can spell today, plus
#: the structural instructions every lane body carries.  Keep in step with
#: ``ssa_glsl_compute_backend`` -- widening that emitter and this set
#: together is how the GPU lane grows.
_GPU_DIALECT_OPERATIONS = frozenset({
    "Fma", "Add", "Sub", "Mul", "Div", "Neg",
    "Const", "const", "Load", "Store", "GetElementPtr",
    "Br", "Ret",
})


@dataclass(frozen=True)
class ComputeLaneVerdict:
    function: str
    region_id: int
    outline_name: str
    eligible: bool
    reasons: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "region_id": self.region_id,
            "outline": self.outline_name,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ComputeSelectionReport:
    verdicts: tuple[ComputeLaneVerdict, ...]

    @property
    def eligible(self) -> tuple[ComputeLaneVerdict, ...]:
        return tuple(v for v in self.verdicts if v.eligible)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema": "turing.deployment-compute-selection.v1",
            "lanes": [verdict.as_record() for verdict in self.verdicts],
        }


def select_compute_lanes(module: Any) -> ComputeSelectionReport:
    """Judge every outlined deployment lane against the GPU dialect."""

    verdicts: list[ComputeLaneVerdict] = []
    outlines = (module.metadata or {}).get("deployment_outlines", {})
    for (function_name, region_id), record in outlines.items():
        outline = module.functions.get(record.outline_name)
        if outline is None:
            verdicts.append(ComputeLaneVerdict(
                str(function_name), int(region_id), record.outline_name,
                False, ("outline function is missing from the module",),
            ))
            continue
        reasons: list[str] = []
        body_blocks = [
            name for name in outline.blocks if name != "lane_return"
        ]
        if len(body_blocks) != 1:
            reasons.append(
                "lane body is not straight-line: blocks "
                f"{body_blocks!r}; GPU lowering needs one block or "
                "predication"
            )
        uncovered: dict[str, int] = {}
        internal_calls: list[str] = []
        for block in outline.blocks.values():
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation in {"Call", "call"}:
                    callee = str(
                        (instruction.attributes or {}).get("callee") or "?"
                    )
                    internal_calls.append(callee)
                    continue
                if operation in {"CondBr", "condbr"}:
                    uncovered["CondBr"] = uncovered.get("CondBr", 0) + 1
                    continue
                if operation not in _GPU_DIALECT_OPERATIONS:
                    uncovered[operation] = uncovered.get(operation, 0) + 1
        if internal_calls:
            unique = sorted(set(internal_calls))
            reasons.append(
                "lane calls internal functions "
                f"{unique[:6]!r}; GPU lowering requires inlining or a "
                "GPU spelling for each"
            )
        if uncovered:
            reasons.append(
                "operations outside the GPU dialect: "
                + ", ".join(
                    f"{op} x{count}" for op, count in sorted(uncovered.items())
                )
            )
        if record.guarded_blocks:
            reasons.append(
                "lane holds an effect-locked shared append; GPU lanes have "
                "no ordered effect primitive"
            )
        verdicts.append(ComputeLaneVerdict(
            str(function_name), int(region_id), record.outline_name,
            not reasons, tuple(reasons),
        ))
    return ComputeSelectionReport(tuple(verdicts))


__all__ = [
    "ComputeLaneVerdict",
    "ComputeSelectionReport",
    "select_compute_lanes",
]
