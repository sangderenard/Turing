import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.cells.bath import SPHAdapter, MACAdapter, HybridAdapter
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams
from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams
from src.cells.bath.hybrid_fluid import HybridFluid, HybridParams
from src.cells.bath.surface_animator import Tileset, TileVariant, SurfaceAnimator


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
    # guard against NaN/inf propagation in solver
    assert np.all(np.isfinite(sample["S"]))
    assert np.all(np.isfinite(sim.pr))


def test_adapter_visualization_state_api():
    """All adapters expose positions, vectors, and surface batches."""
    # SPH
    params = FluidParams()
    sim_sph = DiscreteFluid(
        positions=np.zeros((1, 3)),
        velocities=None,
        temperature=None,
        salinity=None,
        params=params,
    )
    adapter_sph = SPHAdapter(sim_sph)
    state_sph = adapter_sph.visualization_state()
    assert {"positions", "vectors", "surface_batches"} <= set(state_sph)

    # MAC with animator
    vparams = VoxelFluidParams(nx=1, ny=1, nz=1)
    sim_mac = VoxelMACFluid(vparams)
    tileset = Tileset(frames_per_row=1, rows_per_variant=1,
                      variants={0: TileVariant(mesh_id="q", atlas_row=0, fps_base=1.0)})
    anim = SurfaceAnimator(tileset, dx=vparams.dx, dim=3)
    adapter_mac = MACAdapter(sim_mac, animator=anim)
    adapter_mac.step(0.0)
    state_mac = adapter_mac.visualization_state()
    assert {"positions", "vectors", "surface_batches"} <= set(state_mac)
    assert isinstance(state_mac["surface_batches"], list)

    # Hybrid
    hparams = HybridParams(dx=0.05)
    sim_h = HybridFluid(shape=(1, 1, 1), n_particles=0, params=hparams)
    adapter_h = HybridAdapter(sim_h)
    state_h = adapter_h.visualization_state()
    assert {"positions", "vectors", "surface_batches"} <= set(state_h)
