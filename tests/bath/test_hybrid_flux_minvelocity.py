import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_minvelocity_bounds_fluxes():
    params = VoxelFluidParams(nx=2, ny=2, nz=1)
    vf = VoxelMACFluid(params)
    vf.u.fill(10.0)
    vf.v.fill(10.0)
    vf.minvelocity(1.0)
    assert np.all(np.abs(vf.u) <= 1.0)
    assert np.all(np.abs(vf.v) <= 1.0)
