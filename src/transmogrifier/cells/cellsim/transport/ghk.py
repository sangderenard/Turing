from __future__ import annotations
import math
from ..core.units import R as RGAS, F


def ghk_flux(P: float, z: int, Ci: float, Ce: float, Vm: float, T: float, Rgas: float = RGAS, Fconst: float = F) -> float:
    """Goldman–Hodgkin–Katz channel flux (mol/s per m^2).

    P: permeability coefficient
    z: ion valence
    Ci, Ce: inner/outer concentrations
    Vm: membrane potential (volts)
    T: absolute temperature (K)
    """
    if z == 0:
        return P * (Ci - Ce)
    alpha = z * Fconst * Vm / (Rgas * T)
    if abs(alpha) < 1e-6:
        return P * (Ci - Ce)
    num = Ci - Ce * math.exp(-alpha)
    return P * alpha * num / (1.0 - math.exp(-alpha))
