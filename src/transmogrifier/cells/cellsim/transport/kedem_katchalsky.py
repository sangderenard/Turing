"""Transport equations implemented with vectorised NumPy operations."""

from typing import Dict, Iterable, Tuple

import numpy as np

from ..core.units import R as RGAS
from ..core.units import EPS

def arrhenius(P0: float, Ea: float | None, T: float) -> float:
    if Ea is None:
        return P0
    # crude Arrhenius with base e
    return P0 * (2.718281828)**(-Ea/(RGAS*T))

def _arr_from(obj, species: list[str], *, default: float = 0.0) -> np.ndarray:
    """Return a 1D array for ``species`` from ``obj`` which may be a dict or array."""

    if isinstance(obj, dict):
        return np.array([obj.get(sp, default) for sp in species], dtype=float)
    arr = np.asarray(obj, dtype=float)
    if arr.shape != (len(species),):
        raise ValueError("array inputs must match species length")
    return arr


def fluxes(
    comp_left,
    comp_right,
    species: Iterable[str],
    Lp: float,
    Ps: Dict[str, float] | np.ndarray,
    sigma: Dict[str, float] | np.ndarray,
    A: float,
    T: float,
    Rgas: float = RGAS,
    C_left_override: dict | np.ndarray | None = None,
    C_right_override: dict | np.ndarray | None = None,
    Jv_pressure_term: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Return ``(dV_left, dS_left)`` using Kedem–Katchalsky with solvent drag.

    Heavy arithmetic is delegated to NumPy to allow broadcasting across many
    species simultaneously.  The returned species flux is still provided as a
    mapping for compatibility with existing callers.
    """

    species = list(species)
    C_L = (
        comp_left.conc(species) if C_left_override is None else C_left_override
    )
    C_R = (
        comp_right.conc(species) if C_right_override is None else C_right_override
    )

    C_L_arr = _arr_from(C_L, species)
    C_R_arr = _arr_from(C_R, species)
    sigma_arr = _arr_from(sigma, species, default=1.0)
    P_arr = _arr_from(Ps, species)

    # Osmotic term Σ σ_i R T (C_R - C_L)
    osm = np.sum(sigma_arr * Rgas * T * (C_R_arr - C_L_arr))

    Jv = Lp * A * (Jv_pressure_term - osm)  # left->right positive
    dV_L = -Jv

    Js = P_arr * A * (C_R_arr - C_L_arr) + (1.0 - sigma_arr) * C_L_arr * Jv
    dS_L = {sp: -Js[i] for i, sp in enumerate(species)}
    return dV_L, dS_L
