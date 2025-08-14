import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_viscosity_symmetry_and_energy_decay():
    params = FluidParams(viscosity_nu=1e-6, gravity=(0.0, 0.0, 0.0))
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=velocities, temperature=None, salinity=None, params=params)
    fluid._build_grid()
    fluid._compute_density(include_self=True)

    f = fluid._viscosity_forces()
    # forces must be equal and opposite
    assert np.allclose(f[0], -f[1])

    dt = 1e-3
    v_new = fluid.v + dt * f / fluid.m[:, None]

    m = fluid.m
    E0 = 0.5 * np.sum(m * np.sum(fluid.v**2, axis=1))
    E1 = 0.5 * np.sum(m * np.sum(v_new**2, axis=1))

    assert E1 <= E0
