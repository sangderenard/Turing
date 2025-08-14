import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_pressure_projection_uniform_divergence_nullspace():
    params = VoxelFluidParams(nx=2, ny=2, nz=2)
    fluid = VoxelMACFluid(params)

    # Create a velocity field with uniform divergence
    ramp = np.tile(np.arange(params.nx + 1, dtype=float)[:, None, None], (1, params.ny, params.nz))
    fluid.u = ramp
    fluid.v.fill(0.0)
    fluid.w.fill(0.0)

    fluid._project(dt=0.1)

    fluid_cells = ~fluid.solid
    mean_pressure = float(fluid.pr[fluid_cells].mean())

    assert np.all(np.isfinite(fluid.pr))
    assert abs(mean_pressure) < 1e-6
