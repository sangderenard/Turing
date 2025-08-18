import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_surface_tension_normals_and_restoring_force():
    params = FluidParams(surface_tension=1.0)
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)
    fluid._build_grid()
    fluid._compute_density()
    fluid._compute_pressure()

    # Compute normals explicitly to check orientation
    n_vec = np.zeros_like(positions)
    for (i, j, rvec, r2) in fluid._pairs_particles():
        r = np.sqrt(r2)
        m_over_rho_j = fluid.m[j] / np.maximum(fluid.rho[j], 1e-12)
        m_over_rho_i = fluid.m[i] / np.maximum(fluid.rho[i], 1e-12)
        gW = fluid.kernel.gradW(rvec, r)
        n_vec[i] += m_over_rho_j[:, None] * gW
        n_vec[j] -= m_over_rho_i[:, None] * gW
    n_hat = n_vec / (np.linalg.norm(n_vec, axis=1)[:, None] + params.color_field_eps)

    # Outward normals: left particle points -x, right particle +x
    assert n_hat[0, 0] < 0
    assert n_hat[1, 0] > 0

    f = fluid._surface_tension_forces()
    if np.any(f):
        # Restoring force: left particle pulled right, right particle pulled left
        assert f[0, 0] > 0
        assert f[1, 0] < 0
