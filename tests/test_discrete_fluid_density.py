import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_two_particle_density_symmetry():
    params = FluidParams()
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)
    fluid._build_grid()
    fluid._compute_density(include_self=True)

    r = np.linalg.norm(positions[1] - positions[0])
    expected = params.particle_mass * (
        fluid.kernel.W(np.array([0.0]))[0] + fluid.kernel.W(np.array([r]))[0]
    )

    assert np.allclose(fluid.rho[0], fluid.rho[1])
    assert np.allclose(fluid.rho, expected)
