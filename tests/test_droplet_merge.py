import numpy as np
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams

def test_droplet_merge_transfers_mass_and_momentum():
    params = FluidParams(surface_tension=0.05)
    positions = np.zeros((1,3), dtype=float)
    velocities = np.zeros((1,3), dtype=float)
    df = DiscreteFluid(positions, velocities, None, None, params)
    # place a droplet at the particle position heading in +x
    df.droplet_p = np.array([[0.0, 0.0, 0.0]])
    df.droplet_v = np.array([[1.0, 0.0, 0.0]])
    # merge
    df._merge_droplets()
    # droplet removed
    assert df.droplet_p.size == 0
    # mass increased by droplet mass
    expected_mass = df.params.particle_mass + df.droplet_mass
    assert np.allclose(df.m[0], expected_mass)
    assert np.allclose(df.m_target[0], expected_mass)
    # velocity adjusted by momentum conservation
    expected_v = (df.droplet_mass / expected_mass) * np.array([1.0,0.0,0.0])
    assert np.allclose(df.v[0], expected_v)
