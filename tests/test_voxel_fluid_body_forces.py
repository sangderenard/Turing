import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_body_force_shapes_and_solids_unchanged():
    params = VoxelFluidParams(nx=2, ny=2, nz=2, gravity=(1.0, 2.0, 3.0))
    fluid = VoxelMACFluid(params)

    solid = np.zeros((2, 2, 2), dtype=bool)
    solid[0, 0, 0] = True
    fluid.set_solid_mask(solid)

    fluid.u.fill(1.0)
    fluid.v.fill(2.0)
    fluid.w.fill(3.0)

    u0 = fluid.u.copy()
    v0 = fluid.v.copy()
    w0 = fluid.w.copy()

    fluid._add_body_forces(dt=0.1)

    assert fluid.u.shape == (3, 2, 2)
    assert fluid.v.shape == (2, 3, 2)
    assert fluid.w.shape == (2, 2, 3)

    assert np.array_equal(fluid.u[fluid.solid_u], u0[fluid.solid_u])
    assert np.array_equal(fluid.v[fluid.solid_v], v0[fluid.solid_v])
    assert np.array_equal(fluid.w[fluid.solid_w], w0[fluid.solid_w])

    assert np.any(fluid.u[~fluid.solid_u] != u0[~fluid.solid_u])
    assert np.any(fluid.v[~fluid.solid_v] != v0[~fluid.solid_v])
    assert np.any(fluid.w[~fluid.solid_w] != w0[~fluid.solid_w])
