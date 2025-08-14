import numpy as np
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def make_fluid(velocities, salinity=None):
    positions = np.zeros((len(velocities), 3), dtype=float)
    velocities = np.array(velocities, dtype=float)
    if salinity is not None:
        salinity = np.array(salinity, dtype=float)
    params = FluidParams()
    return DiscreteFluid(positions, velocities, None, salinity, params)


def test_droplet_emission_count():
    df = make_fluid([[6, 0, 0], [4, 0, 0], [7, 0, 0]])
    df.emit_droplets(threshold=5.0)
    assert df.droplet_p.shape[0] == 2


def test_mass_solute_conservation_after_reabsorption():
    df = make_fluid([[0, 0, 0]], salinity=[0.1])
    initial_mass = df.m.sum()
    initial_solute = df.solute_mass.sum()
    df.emit_droplets(indices=[0])
    df._merge_droplets()
    final_mass = df.m.sum()
    final_solute = df.solute_mass.sum()
    assert np.isclose(final_mass, initial_mass + df.droplet_mass)
    assert np.isclose(final_solute, initial_solute)


def test_repeated_cycle_stability():
    df = make_fluid([[0, 0, 0]])
    initial_mass = df.m.sum()
    cycles = 5
    for _ in range(cycles):
        df.emit_droplets(indices=[0])
        df._merge_droplets()
    assert df.droplet_p.size == 0
    assert df.droplet_v.size == 0
    expected_mass = initial_mass + cycles * df.droplet_mass
    assert np.isclose(df.m.sum(), expected_mass)
