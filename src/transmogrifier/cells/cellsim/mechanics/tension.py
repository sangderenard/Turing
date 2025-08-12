from ..core.geometry import sphere_area_from_volume

def laplace_pressure(A0, V, elastic_k, visc_eta, eps_prev, dt):
    A, R = sphere_area_from_volume(V)
    eps = (A / A0) - 1.0
    deps_dt = (eps - eps_prev) / max(dt, 1e-18)
    T = elastic_k * eps + visc_eta * deps_dt
    dP_tension = 2.0 * T / max(R, 1e-12)
    return dP_tension, eps
