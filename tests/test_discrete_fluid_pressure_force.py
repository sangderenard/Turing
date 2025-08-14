import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_two_particle_pressure_repulsive():
    params = FluidParams()
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)
    fluid._build_grid()

    # Equal positive pressure for both particles
    fluid.P[:] = 1.0e5

    f = fluid._pressure_forces()

    # Forces must be equal and opposite
    assert np.allclose(f[0], -f[1])
    # And they must be repulsive: particle 0 (left) feels force to the left
    assert f[0][0] < 0
