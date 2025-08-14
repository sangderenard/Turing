import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_discrete_source_conservation():
    params = FluidParams(source_relaxation=1.0)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05],
    ], dtype=np.float64)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)

    initial_mass = fluid.m.sum()
    initial_solute = fluid.solute_mass.sum()

    centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    dM = np.array([0.05, -0.02])
    dS = np.array([0.01, -0.004])

    realized = fluid.apply_sources(centers, dM, dS, radius=0.1)
    fluid.step(1e-3)

    final_mass = fluid.m.sum()
    final_solute = fluid.solute_mass.sum()

    assert np.isclose(final_mass, initial_mass + realized["dM"].sum())
    assert np.isclose(final_solute, initial_solute + realized["dS_mass"].sum())
