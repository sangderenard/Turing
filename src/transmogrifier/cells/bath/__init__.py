from dataclasses import dataclass, field
from typing import Dict, Iterable, List, TYPE_CHECKING

from ..cellsim.transport.kedem_katchalsky import arrhenius, fluxes
from ..cellsim.core.geometry import sphere_area_from_volume
from ..cellsim.core.units import R as RGAS

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..cellsim.data.state import Cell


@dataclass
class Bath:
    r"""External bath surrounding simulated cells.

    Attributes mirror a minimal compartment with bulk thermodynamic
    parameters.  The ``conc`` method mirrors the :class:`Compartment`
    interface used elsewhere in cellsim.

    Notes
    -----
    Water is treated with mild compressibility :math:`\Delta V = \kappa V\Delta P`
    (:math:`\kappa` typically :math:`4.5\times10^{-10}\,\mathrm{Pa^{-1}}`).
    Thermal response follows ``Q = \rho c_p V \Delta T`` with density modelled as
    ``\rho(T) = \rho_0 [1 - \beta (T - T_0)]`` where ``\beta\approx2.07\times10^{-4}``
    ``\text{K}^{-1}`` and ``T_0 = 298.15\,\text{K}``.  Dynamic viscosity uses
    ``\mu(T) = A\cdot10^{B/(T-C)}`` with ``A=2.414\times10^{-5}`` ``\text{Pa·s}``,
    ``B=247.8`` and ``C=140``.
    """

    V: float  # volume (m^3-ish)
    phi: float = 0.0  # electric potential (V), future use
    n: Dict[str, float] = field(default_factory=dict)  # moles by species
    pressure: float = 10
    temperature: float = 298.15
    density: float = 1000.0  # kg/m^3 for water at room temp
    viscosity: float = 1e-3  # Pa·s at room temp
    heat_capacity: float = 4181.3  # J/(kg·K) for water
    compressibility: float = 0.0  # (Pa^-1)
    min_pressure: float | None = None
    max_pressure: float | None = None

    def conc(self, species: List[str]) -> dict:
        V = max(self.V, 1e-18)
        return {sp: self.n.get(sp, 0.0) / V for sp in species}

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    def update_temperature(self, heat: float) -> None:
        """Update temperature from an energy input.

        Parameters
        ----------
        heat:
            Energy added to the bath in joules.  Positive heats the bath.

        Uses ``ΔT = Q / (ρ c_p V)`` and updates density/viscosity using
        simple water models documented in the class notes.
        """

        mass = self.density * max(self.V, 1e-18)
        if mass > 0.0 and self.heat_capacity > 0.0:
            dT = heat / (mass * self.heat_capacity)
            self.temperature += dT

        # Update density (linear thermal expansion around 25°C)
        beta = 2.07e-4  # 1/K
        self.density = 1000.0 * (1 - beta * (self.temperature - 298.15))

        # Update dynamic viscosity via empirical relation
        A = 2.414e-5
        B = 247.8
        C = 140.0
        self.viscosity = A * 10 ** (B / (self.temperature - C))

    # ------------------------------------------------------------------
    # Compressibility helpers
    # ------------------------------------------------------------------
    def cap_fluxes(self, dV: float, dS: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Cap proposed cell volume/solute changes by remaining compressibility.

        Parameters
        ----------
        dV:
            Proposed change in cell volume (bath receives the opposite).
        dS:
            Proposed change in moles for each species (cell perspective).

        Returns
        -------
        dV, dS:
            Possibly reduced volume and solute changes.
        """

        if self.compressibility <= 0.0:
            return dV, dS

        kV = self.compressibility * max(self.V, 1e-18)
        allowed = dV
        if dV > 0.0 and self.min_pressure is not None:
            # bath loses volume → pressure drops
            allowed = min(allowed, (self.pressure - self.min_pressure) * kV)
        if dV < 0.0 and self.max_pressure is not None:
            # bath gains volume → pressure rises
            allowed = max(allowed, (self.pressure - self.max_pressure) * kV)

        if allowed != dV and dV != 0.0:
            scale = allowed / dV
            dS = {sp: val * scale for sp, val in dS.items()}
        return allowed, dS


def update_pressure(bath: Bath, sum_dV: float) -> None:
    """Update bath pressure based on total volume change.

    If bath has finite compressibility ``kappa`` then ``ΔV = kappa · V · ΔP``
    so ``ΔP = ΔV / (kappa · V)``.
    """

    if bath.compressibility and bath.compressibility > 0.0:
        bath.pressure += -(sum_dV / (bath.compressibility * max(bath.V, 1e-18)))
        if bath.min_pressure is not None:
            bath.pressure = max(bath.pressure, bath.min_pressure)
        if bath.max_pressure is not None:
            bath.pressure = min(bath.pressure, bath.max_pressure)


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

    dV_cell, dS_cell = bath.cap_fluxes(dV_cell, dS_cell)

    cell.V = max(cell.V + dV_cell, 0.0)
    bath.V = max(bath.V - dV_cell, 0.0)
    for sp, dS in dS_cell.items():
        cell.n[sp] = max(cell.n.get(sp, 0.0) + dS, 0.0)
        bath.n[sp] = max(bath.n.get(sp, 0.0) - dS, 0.0)

    update_pressure(bath, dV_cell)
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

    sum_dV = 0.0
    for cell, dV, dS in zip(cells, dV_cells, dS_cells):
        dV, dS = bath.cap_fluxes(dV, dS)
        cell.V = max(cell.V + dV, 0.0)
        bath.V = max(bath.V - dV, 0.0)
        for sp, dS_val in dS.items():
            cell.n[sp] = max(cell.n.get(sp, 0.0) + dS_val, 0.0)
            bath.n[sp] = max(bath.n.get(sp, 0.0) - dS_val, 0.0)
        sum_dV += dV

    update_pressure(bath, sum_dV)


__all__ = ["Bath", "update_pressure", "kedem_katchalsky_step", "apply_fluxes"]
