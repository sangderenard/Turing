"""Canonical packed state layout for compiled chemistry pieces."""

from __future__ import annotations

from dataclasses import dataclass


CHEMISTRY_PRECISION_LIMBS = 2


@dataclass(frozen=True)
class ChemistryPieceLayout:
    """Physical layout of one cell-major precision inventory span.

    Logical index ``cell * state_count + state`` is independent of physical
    precision.  At the LLVM boundary each logical element owns ``limbs``
    adjacent binary64 values, high limb first.
    """

    states: tuple[str, ...]
    cells: int
    limbs: int = CHEMISTRY_PRECISION_LIMBS

    @property
    def logical_size(self) -> int:
        return int(self.cells) * len(self.states)

    @property
    def physical_size(self) -> int:
        return self.logical_size * int(self.limbs)

    def logical_index(self, cell: int, state: str) -> int:
        if not 0 <= int(cell) < int(self.cells):
            raise IndexError(cell)
        try:
            state_index = self.states.index(str(state))
        except ValueError as error:
            raise KeyError(state) from error
        return int(cell) * len(self.states) + state_index

    def physical_index(self, cell: int, state: str, limb: int = 0) -> int:
        if not 0 <= int(limb) < int(self.limbs):
            raise IndexError(limb)
        return self.logical_index(cell, state) * int(self.limbs) + int(limb)


__all__ = ["CHEMISTRY_PRECISION_LIMBS", "ChemistryPieceLayout"]
