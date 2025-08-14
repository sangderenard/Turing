from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Bath:
    """External bath surrounding simulated cells.

    Attributes mirror a minimal compartment with bulk thermodynamic
    parameters.  The ``conc`` method mirrors the Compartment interface
    used elsewhere in cellsim.
    """

    V: float  # volume (m^3-ish)
    phi: float = 0.0  # electric potential (V), future use
    n: Dict[str, float] = field(default_factory=dict)  # moles by species
    pressure: float = 10
    temperature: float = 298.15
    compressibility: float = 0.0

    def conc(self, species: List[str]) -> dict:
        V = max(self.V, 1e-18)
        return {sp: self.n.get(sp, 0.0) / V for sp in species}


def update_pressure(bath: Bath, sum_dV: float) -> None:
    """Update bath pressure based on total volume change.

    If bath has finite compressibility ``kappa`` then ``ΔV = kappa · V · ΔP``
    so ``ΔP = ΔV / (kappa · V)``.
    """

    if bath.compressibility and bath.compressibility > 0.0:
        bath.pressure += -(sum_dV / (bath.compressibility * max(bath.V, 1e-18)))


__all__ = ["Bath", "update_pressure"]
