from __future__ import annotations
from typing import List, Protocol, TypedDict, runtime_checkable

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


@runtime_checkable
class MechanicsProvider(Protocol):
    """Duck-typed 0D mechanics provider contract.

    Implementations may internally simulate higher-D state but expose 0D aggregates.
    """

    def sync(self, cells: List[Cell], bath: Bath) -> None:
        """Sync the provider's internal state from cellsim Cells/Bath before stepping."""
        ...

    # Optional array-first fast-path to avoid Python object conversions
    def sync_arrays(self, *, V, elastic_k, imp, bath_pressure: float) -> None:  # type: ignore[override]
        """Sync provider directly from numpy arrays (optional)."""
        ...

    def step(self, dt: float) -> MechanicsSnapshot:
        """Advance provider by dt and return a snapshot of mechanics aggregates."""
        ...
