import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.cells.bath import SPHAdapter, MACAdapter
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams
from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams


def test_sph_adapter_deposit_and_sample():
    params = FluidParams()
    sim = DiscreteFluid(
        positions=np.zeros((1, 3)),
        velocities=None,
        temperature=None,
        salinity=None,
        params=params,
    )
    adapter = SPHAdapter(sim)

    centers = np.zeros((1, 3))
    res = adapter.deposit(centers, dV=np.array([1e-6]), dS=np.array([0.0]), radius=params.smoothing_length)
    assert "dM" in res and res["dM"].shape == (1,)

    adapter.step(1e-4)
    sample = adapter.sample(centers)
    assert "rho" in sample and sample["rho"].shape == (1,)


def test_mac_adapter_deposit_salinity():
    params = VoxelFluidParams(nx=2, ny=2, nz=2)
    sim = VoxelMACFluid(params)
    adapter = MACAdapter(sim)

    centers = np.array([[params.dx * 0.5, params.dx * 0.5, params.dx * 0.5]])
    adapter.deposit(centers, dV=np.array([0.0]), dS=np.array([0.5]), radius=params.dx)
    assert np.any(sim.S > 0.0)

    adapter.step(1e-4)
    sample = adapter.sample(centers)
    assert "S" in sample and sample["S"].shape == (1,)
