import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_diffuse_cc_explicit_handles_degenerate_dims():
    """_diffuse_cc_explicit should support 2D and 1D grids without errors."""
    for ny, nz in [(4, 1), (1, 1)]:
        params = VoxelFluidParams(nx=5, ny=ny, nz=nz)
        fluid = VoxelMACFluid(params)
        F = np.ones((params.nx, params.ny, params.nz))
        out = fluid._diffuse_cc_explicit(F, kappa=1e-4, dt=0.1)
        assert out.shape == F.shape
        assert np.allclose(out, F)
