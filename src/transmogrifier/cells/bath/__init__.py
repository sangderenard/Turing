from dataclasses import dataclass, field
from typing import Dict, Iterable, List, TYPE_CHECKING

from ..cellsim.transport.kedem_katchalsky import arrhenius, fluxes
from ..cellsim.core.geometry import sphere_area_from_volume
from ..cellsim.core.units import R as RGAS

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..cellsim.data.state import Cell


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




def kedem_katchalsky_step(
    cell: "Cell",
    bath: Bath,
    species: Iterable[str],
    *,
    area: float | None = None,
    T: float | None = None,
    Rgas: float = RGAS,
) -> float:
    """Exchange volume and solute between a cell and the bath.

    This is a thin wrapper around the vectorised Kedem–Katchalsky ``fluxes``
    routine used by the main cellsim engine.  It computes permeabilities via
    Arrhenius activation energies, applies equal-and-opposite updates to the
    cell and bath, and returns the cell volume change ``dV``.
    """

    species = list(species)
    T = T if T is not None else bath.temperature
    A = area if area is not None else sphere_area_from_volume(cell.V)[0]

    Lp = arrhenius(cell.Lp0, cell.Ea_Lp, T)
    Ps = {sp: arrhenius(cell.Ps0.get(sp, 0.0), cell.Ea_Ps.get(sp), T) for sp in species}
    sigma = {sp: cell.sigma.get(sp, 1.0) for sp in species}
    Jv_term = getattr(cell, "base_pressure", 0.0) - bath.pressure

    dV_cell, dS_cell = fluxes(
        comp_left=cell,
        comp_right=bath,
        species=species,
        Lp=Lp,
        Ps=Ps,
        sigma=sigma,
        A=A,
        T=T,
        Rgas=Rgas,
        Jv_pressure_term=Jv_term,
    )

    cell.V = max(cell.V + dV_cell, 0.0)
    bath.V = max(bath.V - dV_cell, 0.0)
    for sp, dS in dS_cell.items():
        cell.n[sp] = max(cell.n.get(sp, 0.0) + dS, 0.0)
        bath.n[sp] = max(bath.n.get(sp, 0.0) - dS, 0.0)

    return dV_cell


def apply_fluxes(
    cells: Iterable["Cell"],
    bath: Bath,
    dV_cells: Iterable[float],
    dS_cells: Iterable[Dict[str, float]],
) -> None:
    """Apply precomputed volume and solute fluxes.

    Parameters
    ----------
    cells:
        Iterable of cells whose state will be updated.
    bath:
        Bath to receive equal-and-opposite updates.
    dV_cells:
        Sequence of volume changes for each cell (positive means cell gains volume).
    dS_cells:
        Sequence of dictionaries mapping species to mole changes per cell.
    """

    for cell, dV, dS in zip(cells, dV_cells, dS_cells):
        cell.V = max(cell.V + dV, 0.0)
        bath.V = max(bath.V - dV, 0.0)
        for sp, dS_val in dS.items():
            cell.n[sp] = max(cell.n.get(sp, 0.0) + dS_val, 0.0)
            bath.n[sp] = max(bath.n.get(sp, 0.0) - dS_val, 0.0)


__all__ = ["Bath", "kedem_katchalsky_step", "apply_fluxes"]
