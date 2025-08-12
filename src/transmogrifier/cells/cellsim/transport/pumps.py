from __future__ import annotations
from typing import Dict


def na_k_atpase_constant(J_pump_cell: float) -> float:
    """Return pump mol/s (cell-wide). Positive means: 3 Na out, 2 K in."""
    return max(J_pump_cell, 0.0)


def na_k_atpase_saturating(
    C_Nai: float,
    C_Ko: float,
    A: float,
    *,
    Jmax: float = 0.0,
    Km_Nai: float = 10.0,
    Km_Ko: float = 1.5,
    eps: float = 0.0,
    alpha_tension: float = 0.0,
) -> float:
    """Saturating surrogate of Na/K ATPase.

    J = Jmax * (Nai/(Km_Nai+Nai))^3 * (Ko/(Km_Ko+Ko))^2
        * (1 + alpha_tension*max(eps,0)) * A

    Returns mol/s across the membrane (cell-wide).
    """
    if Jmax <= 0.0 or A <= 0.0:
        return 0.0
    f_Nai = (C_Nai / (Km_Nai + C_Nai)) if C_Nai > 0 else 0.0
    f_Ko = (C_Ko / (Km_Ko + C_Ko)) if C_Ko > 0 else 0.0
    boost = 1.0 + alpha_tension * max(eps, 0.0)
    return Jmax * (f_Nai ** 3) * (f_Ko ** 2) * boost * A


def apply_na_k_pump_to_left_changes(dS_left: Dict[str, float], J_pump: float, dt: float) -> None:
    """Apply pump flux to left-compartment species changes (mol)."""
    if J_pump <= 0.0 or dt <= 0.0:
        return
    dS_left["Na"] = dS_left.get("Na", 0.0) - 3.0 * J_pump * dt
    dS_left["K"] = dS_left.get("K", 0.0) + 2.0 * J_pump * dt


def na_k_pump_flux(comp_left, comp_right, *, A: float, rate: float, dt: float) -> Dict[str, float]:
    """Backward-compatible constant-rate pump helper.

    Computes cell-wide pump flux from a surface rate (mol/(m^2*s)).
    Returns species changes for the left compartment (cell).
    """
    if rate <= 0.0 or dt <= 0.0 or A <= 0.0:
        return {}
    dS: Dict[str, float] = {}
    J = na_k_atpase_constant(rate * A)
    apply_na_k_pump_to_left_changes(dS, J, dt)
    return dS

