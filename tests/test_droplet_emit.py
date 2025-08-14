import numpy as np
from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams

def make_fluid(velocities):
    positions = np.zeros((len(velocities), 3), dtype=float)
    velocities = np.array(velocities, dtype=float)
    params = FluidParams()
    return DiscreteFluid(positions, velocities, None, None, params)

def test_emit_droplets_threshold():
    df = make_fluid([[0,0,0],[6,0,0]])
    df.emit_droplets(threshold=5.0)
    assert df.droplet_p.shape[0] == 1
    assert np.allclose(df.droplet_v[0], [6,0,0])

def test_droplet_integration_gravity_drag():
    df = make_fluid([[1,0,0]])
    df.droplet_drag = 0.5
    df.emit_droplets(indices=[0])
    df._substep(0.1)
    expected_v = np.array([1.0,0.0,0.0]) + 0.1*(np.array(df.params.gravity) - 0.5*np.array([1.0,0.0,0.0]))
    expected_p = 0.1*expected_v
    assert np.allclose(df.droplet_v[0], expected_v)
    assert np.allclose(df.droplet_p[0], expected_p)
