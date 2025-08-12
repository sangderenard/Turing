"""Inner organelle ↔ cytosol exchange using vectorised math."""

from typing import Iterable

import numpy as np

from ..core.geometry import sphere_area_from_volume
from ..transport.kedem_katchalsky import arrhenius, fluxes
from ..core.units import R as RGAS
from tqdm.auto import tqdm  # type: ignore

def cytosol_free_volume(cell) -> float:
    occ = sum(getattr(o, "V_solid", 0.0) + o.V_lumen() for o in getattr(cell, "organelles", []))
    return max(cell.V - occ, 1e-18)

def inner_exchange(cell, T: float, dt: float, species: Iterable[str], Rgas: float = RGAS):
    """Cytosol ↔ organelle lumen exchange for one cell.
    - Uses excluded volume for cytosolic concentrations (V_free).
    - Conserves cell total volume and moles in this inner step.
    - Updates organelle lumen volume only.
    """
    # cytosolic "compartment" view with free volume override
    species = list(species)
    V_free = cytosol_free_volume(cell)
    n_cyt = np.array([cell.n.get(sp, 0.0) for sp in species], dtype=float)
    Ccyt = n_cyt / V_free

    for o in tqdm(getattr(cell, "organelles", []), desc="organelles", leave=False):
        if getattr(o, "incompressible", False) or o.V_lumen() <= 0.0:
            continue
        V_lum = max(o.V_lumen(), 1e-18)
        A_o, _ = sphere_area_from_volume(V_lum)

        n_org = np.array([o.n.get(sp, 0.0) for sp in species], dtype=float)
        Corg = n_org / V_lum

        # tension coupling proxy from overall cell strain could be added upstream; keep mild here
        Lp = arrhenius(o.Lp0, o.Ea_Lp, T)
        Ps = np.array([
            arrhenius(o.Ps0.get(sp, 0.0), o.Ea_Ps.get(sp), T) for sp in species
        ], dtype=float)
        sigma = np.array([o.sigma.get(sp, 1.0) for sp in species], dtype=float)

        dV_cyt, dS_cyt = fluxes(
            comp_left=cell,
            comp_right=o,
            species=species,
            Lp=Lp,
            Ps=Ps,
            sigma=sigma,
            A=A_o,
            T=T,
            Rgas=Rgas,
            C_left_override=Ccyt,
            C_right_override=Corg,
            Jv_pressure_term=0.0,
        )

        # Apply equal & opposite to conserve totals
        dV_lum = -dV_cyt
        o.set_V_lumen(max(V_lum + dV_lum, 0.0))

        for sp, dS in tqdm(dS_cyt.items(), desc="species", leave=False):
            cell.n[sp] = max(cell.n.get(sp, 0.0) + dS, 0.0)
            o.n[sp] = max(o.n.get(sp, 0.0) - dS, 0.0)
