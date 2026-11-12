import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


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


def test_overlapping_support_accumulates_every_particle_and_field():
    positions = np.array([[0., 0., 0.], [.05, 0., 0.], [.1, .02, 0.]])
    params = FluidParams(particle_mass=1., smoothing_length=.2)
    fluid = DiscreteFluid(positions, None, None, None, params)
    fluid.m[:] = [1., 2., 3.]
    fluid.rho[:] = [900., 1000., 1100.]
    fluid.P[:] = [1., 5., 9.]
    fluid.T[:] = [280., 300., 320.]
    fluid.S[:] = [.01, .02, .04]
    fluid.v[:] = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    queries = np.array([[.025, 0., 0.], [.08, .01, 0.]])
    before = {name: getattr(fluid, name).copy() for name in ('p', 'm', 'rho', 'P', 'T', 'S', 'v')}
    sampled = fluid.sample_at(queries)
    for i, query in enumerate(queries):
        kernel = fluid.kernel.W(np.linalg.norm(positions - query, axis=1))
        weights = kernel * fluid.m / fluid.rho
        np.testing.assert_allclose(sampled['rho'][i], np.sum(fluid.m * kernel))
        for key in ('P', 'T', 'S', 'v'):
            values = getattr(fluid, key)
            expected = np.sum(values * (weights[:, None] if key == 'v' else weights), axis=0) / weights.sum()
            np.testing.assert_allclose(sampled[key][i], expected)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(fluid, name), values)

    # Reordering identity/storage cannot change the reconstructed field.
    order = np.array([2, 0, 1])
    for name in before:
        setattr(fluid, name, before[name][order].copy())
    permuted = fluid.sample_at(queries)
    for key in sampled:
        np.testing.assert_allclose(permuted[key], sampled[key])


def test_empty_support_is_zero_and_empty_query_has_no_samples():
    params = FluidParams(particle_mass=1., smoothing_length=.2)
    for positions in (np.zeros((0, 3)), np.zeros((1, 3))):
        fluid = DiscreteFluid(positions, None, None, None, params)
        outside = fluid.sample_at(np.array([[1., 0., 0.]]))
        for values in outside.values():
            np.testing.assert_array_equal(values, np.zeros_like(values))
        empty = fluid.sample_at(np.empty((0, 3)))
        for key, values in empty.items():
            assert values.shape == ((0, 3) if key == 'v' else (0,))
