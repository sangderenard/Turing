from ..data.state import Bath

def update_pressure(bath: Bath, sum_dV: float):
    if bath.compressibility and bath.compressibility > 0.0:
        bath.pressure += -bath.compressibility * (sum_dV / max(bath.V, 1e-18))
