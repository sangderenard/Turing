from ..data.state import Bath

def update_pressure(bath: Bath, sum_dV: float):
    if bath.compressibility and bath.compressibility > 0.0:
        # compressibility κ: ΔV = κ·V·ΔP  ⇒  ΔP = ΔV / (κ·V)
        bath.pressure += -(sum_dV / (bath.compressibility * max(bath.V, 1e-18)))
