import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_uniform_scalar_interpolation_with_density_variation():
    rng = np.random.default_rng(0)
    n_particles = 20
    positions = rng.uniform(-0.5, 0.5, size=(n_particles, 3))
    params = FluidParams(particle_mass=1.0, smoothing_length=1.0)
    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)

    # Assign non-uniform densities
    fluid.rho = rng.uniform(0.5, 2.0, size=n_particles) * params.rest_density

    # Particles carry uniform scalar fields
    fluid.P[:] = 1.0
    fluid.T[:] = 1.0
    fluid.S[:] = 1.0
    fluid.v[:] = 1.0

    # Sample at random points in the domain
    sample_points = rng.uniform(-0.5, 0.5, size=(5, 3))
    out = fluid.sample_at(sample_points)

    assert np.allclose(out['P'], 1.0, atol=1e-6)
    assert np.allclose(out['T'], 1.0, atol=1e-6)
    assert np.allclose(out['S'], 1.0, atol=1e-6)
    assert np.allclose(out['v'], np.ones_like(out['v']), atol=1e-6)
