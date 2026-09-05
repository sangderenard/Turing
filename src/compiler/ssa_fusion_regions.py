"""Define the only legal input boundary for future numerical fusion.

Fusion is deliberately downstream.  Source graphs and control lowering retain
their complete structure.  Only already-lowered repository SSA may be divided
into maximal contiguous, straight-line, effect-free regions, and a backend may
then choose to fuse one of those regions.  This module discovers candidates;
it performs no rewrite and is not on the compilation hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..transmogrifier.ssa import Function, Instr


CONTROL_BARRIERS = frozenset({
    "Ret", "Br", "CondBr", "Switch", "Phi", "Call", "Invoke",
})
EFFECT_BARRIERS = frozenset({
    "Store", "SetAttr", "SetItem", "Publish", "Raise", "Yield",
})


@dataclass(frozen=True, slots=True)
class ContiguousSSARegion:
    function: str
    block: str
    ordinal: int
    instructions: tuple[Instr, ...]

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(str(instruction.op) for instruction in self.instructions)


def discover_contiguous_ssa_regions(
    function: Function,
    fusible_operations: Iterable[str],
) -> tuple[ContiguousSSARegion, ...]:
    """Return maximal SSA runs bounded by control, effects, and unsupported ops."""

    allowed = frozenset(map(str, fusible_operations))
    regions: list[ContiguousSSARegion] = []
    for block_name, block in function.blocks.items():
        current: list[Instr] = []

        def flush() -> None:
            if current:
                regions.append(ContiguousSSARegion(
                    function.name,
                    str(block_name),
                    len(regions),
                    tuple(current),
                ))
                current.clear()

        for instruction in block.instrs:
            operation = str(instruction.op)
            effectful = bool(instruction.attributes.get("effectful", False))
            if (
                operation in CONTROL_BARRIERS
                or operation in EFFECT_BARRIERS
                or operation not in allowed
                or effectful
            ):
                flush()
                continue
            current.append(instruction)
        flush()
    return tuple(regions)


__all__ = [
    "CONTROL_BARRIERS",
    "EFFECT_BARRIERS",
    "ContiguousSSARegion",
    "discover_contiguous_ssa_regions",
]
