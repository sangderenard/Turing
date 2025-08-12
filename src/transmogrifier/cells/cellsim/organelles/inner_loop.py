from typing import Iterable
from ..core.geometry import sphere_area_from_volume
from ..transport.kedem_katchalsky import arrhenius, fluxes
from ..core.units import R as RGAS

def cytosol_free_volume(cell) -> float:
    occ = sum(o.volume_total for o in getattr(cell, "organelles", []))
    return max(cell.V - occ, 1e-18)

def inner_exchange(cell, T: float, dt: float, species: Iterable[str], Rgas: float = RGAS):
    """Cytosol ↔ organelle lumen exchange for one cell.
    - Uses excluded volume for cytosolic concentrations (V_free).
    - Conserves cell total volume and moles in this inner step.
    - Updates organelle volume_total (via lumen change).
    """
    # cytosolic "compartment" view with free volume override
    V_free = cytosol_free_volume(cell)
    Ccyt = {sp: (cell.n.get(sp,0.0)/V_free) for sp in species}

    for o in getattr(cell, "organelles", []):
        V_lum = max(o.V_lumen(), 1e-18)
        A_o, R_o = sphere_area_from_volume(V_lum)

        Corg = {sp: (o.n.get(sp,0.0)/V_lum) for sp in species}

        # tension coupling proxy from overall cell strain could be added upstream; keep mild here
        Lp = arrhenius(o.Lp0, o.Ea_Lp, T)
        Ps = {sp: arrhenius(o.Ps0.get(sp,0.0), o.Ea_Ps.get(sp), T) for sp in species}
        sigma = {sp: o.sigma.get(sp,1.0) for sp in species}

        # zero hydrostatic across organelle by default
        dV_cyt, dS_cyt = fluxes(
            comp_left=cell, comp_right=o,
            species=species,
            Lp=Lp, Ps=Ps, sigma=sigma, A=A_o, T=T, Rgas=Rgas,
            C_left_override=Ccyt, C_right_override=Corg,
            Jv_pressure_term=0.0
        )
        # Apply equal&opposite to conserve cell totals
        # left is cytosol: dV_cyt is change in cytosol volume; we translate into organelle lumen change
        dV_lum = -dV_cyt  # lumen gains what cytosol loses
        o.volume_total = max(o.volume_total + dV_lum, 1e-18)

        # species
        for sp, dS in dS_cyt.items():
            # cytosol change is dS_cyt[sp]; organelle opposite
            cell.n[sp] = max(cell.n.get(sp,0.0) + dS, 0.0)
            o.n[sp]    = max(o.n.get(sp,0.0)    - dS, 0.0)
