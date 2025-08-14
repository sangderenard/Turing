import os
import sys
import numpy as np
import pytest


# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_salinity_clamp():
    params = VoxelFluidParams(nx=3, ny=3, nz=3, solute_diffusivity=1.0e-9)
    fluid = VoxelMACFluid(params)
    fluid.S = np.linspace(-0.1, 1.1, fluid.S.size).reshape(fluid.S.shape)
    with pytest.warns(RuntimeWarning):
        fluid.step(1e-3)

    assert fluid.S.min() >= 0.0
    assert fluid.S.max() <= 1.0

