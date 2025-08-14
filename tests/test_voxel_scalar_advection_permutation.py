import os
import sys
import itertools
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_scalar_advection_invariance_under_axis_permutation():
    params = VoxelFluidParams(nx=2, ny=2, nz=2)
    fluid = VoxelMACFluid(params)

    rng = np.random.default_rng(0)
    fluid.u = rng.random(fluid.u.shape)
    fluid.v = rng.random(fluid.v.shape)
    fluid.w = rng.random(fluid.w.shape)
    scalar = rng.random((fluid.nx, fluid.ny, fluid.nz))

    u0 = fluid.u.copy()
    v0 = fluid.v.copy()
    w0 = fluid.w.copy()
    F0 = scalar.copy()

    dt = 0.1
    ref = fluid._advect_scalar_cc(F0, dt)

    comps = [u0, v0, w0]
    perms = list(itertools.permutations([0, 1, 2]))
    results = []
    for perm in perms:
        sim = VoxelMACFluid(params)
        sim.u = comps[perm[0]].transpose(perm)
        sim.v = comps[perm[1]].transpose(perm)
        sim.w = comps[perm[2]].transpose(perm)
        Fp = F0.transpose(perm)
        adv = sim._advect_scalar_cc(Fp, dt)
        inv = np.argsort(perm)
        results.append(adv.transpose(inv))

    for r in results[1:]:
        assert np.allclose(r, results[0])
    assert np.allclose(results[0], ref)
