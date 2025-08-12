from typing import Dict

def na_k_pump_flux(comp_left, comp_right, *, A: float, rate: float, dt: float) -> Dict[str,float]:
    """Return dS_left for a Na/K-ATPase pumping 3 Na out, 2 K in.

    rate: mol/(m^2*s) maximum turnover. Positive rate ejects Na from left (cell).
    A: membrane area in m^2.
    dt: timestep in s.
    """
    if rate <= 0.0 or dt <= 0.0 or A <= 0.0:
        return {}
    J = rate * A * dt
    return {"Na": -3.0 * J, "K": 2.0 * J}
