import os
import sys
import itertools
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_velocity_advection_order_invariance():
    params = VoxelFluidParams(nx=2, ny=2, nz=2)
    fluid = VoxelMACFluid(params)

    rng = np.random.default_rng(0)
    fluid.u = rng.random(fluid.u.shape)
    fluid.v = rng.random(fluid.v.shape)
    fluid.w = rng.random(fluid.w.shape)

    u0 = fluid.u.copy()
    v0 = fluid.v.copy()
    w0 = fluid.w.copy()
    dt = 0.1

    def advect(order):
        fluid.u[:] = u0
        fluid.v[:] = v0
        fluid.w[:] = w0
        for comp in order:
            if comp == 'u':
                fluid.u = fluid._advect_component_face(u0, u0, v0, w0, dt, axis=0)
            elif comp == 'v':
                fluid.v = fluid._advect_component_face(v0, u0, v0, w0, dt, axis=1)
            elif comp == 'w':
                fluid.w = fluid._advect_component_face(w0, u0, v0, w0, dt, axis=2)
        return fluid.u.copy(), fluid.v.copy(), fluid.w.copy()

    orders = list(itertools.permutations(['u', 'v', 'w']))
    results = [advect(order) for order in orders]

    for r in results[1:]:
        assert np.allclose(r[0], results[0][0])
        assert np.allclose(r[1], results[0][1])
        assert np.allclose(r[2], results[0][2])
