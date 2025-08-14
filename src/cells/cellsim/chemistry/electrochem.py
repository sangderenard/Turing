from __future__ import annotations
from typing import Dict
from ..core.units import F


def update_voltage(Vm: float, currents: Dict[str, float], Cm: float, dt: float) -> float:
    """Integrate membrane voltage using C_m dV/dt = -Σ I.

    Vm: membrane potential (V)
    currents: mapping channel/pump -> current (A), positive outward
    Cm: membrane capacitance (F)
    dt: timestep (s)
    """
    if Cm <= 0.0:
        return Vm
    I_total = sum(currents.values())
    return Vm + (-I_total / Cm) * dt
