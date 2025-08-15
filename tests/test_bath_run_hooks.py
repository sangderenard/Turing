import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.cells.bath import (
    SPHAdapter,
    MACAdapter,
    HybridAdapter,
    run_headless,
    run_opengl,
)
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams
from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams
from src.cells.bath.hybrid_fluid import HybridFluid, HybridParams


def _make_sph_adapter():
    params = FluidParams()
    sim = DiscreteFluid(
        positions=np.zeros((1, 3)),
        velocities=None,
        temperature=None,
        salinity=None,
        params=params,
    )
    return SPHAdapter(sim)


def _make_mac_adapter():
    vparams = VoxelFluidParams(nx=1, ny=1, nz=1)
    sim = VoxelMACFluid(vparams)
    return MACAdapter(sim)


def _make_hybrid_adapter():
    hparams = HybridParams(dx=0.05)
    sim = HybridFluid(shape=(1, 1, 1), n_particles=1, params=hparams)
    return HybridAdapter(sim)


def test_run_headless_collects_states():
    adapter = _make_sph_adapter()
    frames = run_headless(adapter, steps=2, dt=1e-4)
    assert len(frames) == 2
    assert all("positions" in f for f in frames)


def test_run_opengl_draw_modes():
    # SPH: points only
    sph = _make_sph_adapter()
    frames = run_opengl(sph, steps=1, dt=0.0, draw="points")
    assert "points" in frames[0] and "vectors" not in frames[0]

    # MAC: vectors only
    mac = _make_mac_adapter()
    frames_mac = run_opengl(mac, steps=1, dt=0.0, draw="vectors")
    assert "vectors" in frames_mac[0] and "points" not in frames_mac[0]

    # Hybrid: both with low alpha
    hyb = _make_hybrid_adapter()
    frames_h = run_opengl(hyb, steps=1, dt=0.0, draw="points+vectors")
    assert "points" in frames_h[0] and "vectors" in frames_h[0]
    colors = frames_h[0]["vectors"]["color"]
    assert np.all(colors[:, 3] <= 0.25 + 1e-6)

