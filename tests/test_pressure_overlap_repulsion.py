import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_pressure_repulsion_overlapping_particles():
    params = FluidParams()
    # Two particles almost at the same location
    positions = np.array([[0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)
    fluid._build_grid()
    # apply equal positive pressure
    fluid.P[:] = 1.0e5
    f = fluid._pressure_forces()
    # Forces equal and opposite
    assert np.allclose(f[0], -f[1])
    # Repulsive along x-axis: particle 0 feels force to the left
    assert f[0][0] < 0
    assert f[1][0] > 0
