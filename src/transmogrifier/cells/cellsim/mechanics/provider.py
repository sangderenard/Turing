from __future__ import annotations
from typing import List, Protocol, TypedDict

from ..data.state import Cell, Bath


class MechanicsSnapshot(TypedDict, total=False):
    """Provider output for one mechanics step.

    keys:
      - pressures: list[float]         per-cell absolute internal pressure estimate
      - areas:     list[float]         optional per-cell membrane surface area to use for fluxes
      - volumes:   list[float]         optional per-cell volumes (informational)
    """

    pressures: List[float]
    areas: List[float]
    volumes: List[float]


class MechanicsProvider(Protocol):
    """Duck-typed 0D mechanics provider contract.

    Implementations may internally simulate higher-D state but expose 0D aggregates.
    """

    def sync(self, cells: List[Cell], bath: Bath) -> None:
        """Sync the provider's internal state from cellsim Cells/Bath before stepping."""
        ...

    def step(self, dt: float) -> MechanicsSnapshot:
        """Advance provider by dt and return a snapshot of mechanics aggregates."""
        ...
