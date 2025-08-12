from typing import Dict, Iterable, Tuple
from ..core.units import R as RGAS
from ..core.units import EPS

def arrhenius(P0: float, Ea: float | None, T: float) -> float:
    if Ea is None:
        return P0
    # crude Arrhenius with base e
    return P0 * (2.718281828)**(-Ea/(RGAS*T))

def fluxes(comp_left, comp_right, species: Iterable[str], Lp: float, Ps: Dict[str,float], sigma: Dict[str,float], A: float, T: float, Rgas: float = RGAS, C_left_override: dict | None = None, C_right_override: dict | None = None, Jv_pressure_term: float = 0.0) -> Tuple[float, Dict[str,float]]:
    """Return (dV_left, dS_left) using Kedem–Katchalsky with solvent drag.
    comp_left/right: have V and n[sp].
    If concentration overrides are provided, use them (for cytosol free-volume case).
    Jv_pressure_term is (P_right - P_left) if caller wants hydrostatic contribution.
    """
    V_L = max(comp_left.V, 1e-18); V_R = max(comp_right.V, 1e-18)
    C_L = (comp_left.conc(list(species)) if C_left_override is None else C_left_override)
    C_R = (comp_right.conc(list(species)) if C_right_override is None else C_right_override)

    # Osmotic term Σ σ_i R T (C_R - C_L)
    osm = 0.0
    for sp in species:
        s = sigma.get(sp, 1.0)
        osm += s * Rgas * T * (C_R[sp] - C_L[sp])

    Jv = Lp * A * (Jv_pressure_term - osm)  # volume flux left->right positive if pressure/osm pushes that way
    dV_L = -Jv  # left volume decreases if flux to right is positive

    dS_L: Dict[str,float] = {}
    for sp in species:
        P = Ps.get(sp, 0.0)
        s = sigma.get(sp, 1.0)
        # solvent drag uses donor conc; pick left as donor for left->right sign convention
        Js = P * A * (C_R[sp] - C_L[sp]) + (1.0 - s) * C_L[sp] * Jv
        dS_L[sp] = -Js  # species move with Js from L to R; left loses Js
    return dV_L, dS_L
