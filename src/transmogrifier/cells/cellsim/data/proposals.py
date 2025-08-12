from dataclasses import dataclass

@dataclass
class CellProposal:
    cell_id: int | None = None
    left: int | None = None
    right: int | None = None
    leftmost: int | None = None
    rightmost: int | None = None
    salinity: float | None = None
    pressure: float | None = None
