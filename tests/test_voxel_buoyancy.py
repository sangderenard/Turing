import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_uniform_gravity_and_solid_faces():
    params = VoxelFluidParams(nx=3, ny=3, nz=3, gravity=(1.0, 2.0, 3.0))
    fluid = VoxelMACFluid(params)

    solid = np.zeros((3, 3, 3), dtype=bool)
    solid[0, 0, 0] = True
    fluid.set_solid_mask(solid)

    dt = 0.1
    fluid._add_body_forces(dt)

    gx, gy, gz = params.gravity
    assert np.allclose(fluid.u[~fluid.solid_u], dt * gx)
    assert np.allclose(fluid.v[~fluid.solid_v], dt * gy)
    assert np.allclose(fluid.w[~fluid.solid_w], dt * gz)

    assert np.allclose(fluid.u[fluid.solid_u], 0.0)
    assert np.allclose(fluid.v[fluid.solid_v], 0.0)
    assert np.allclose(fluid.w[fluid.solid_w], 0.0)
