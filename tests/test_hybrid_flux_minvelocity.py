import numpy as np

from src.cells.bath.hybrid_fluid import HybridFluid, HybridParams


def test_flux_emission_mass_conservation():
    params = HybridParams(
        dx=1.0,
        particle_mass=0.5,
        emit_fraction=0.5,
        p_low=0.0,
        phi_full=0.9,
    )
    sim = HybridFluid(shape=(2, 1, 1), n_particles=0, params=params)
    # Source cell full, target low pressure
    sim.phi[0, 0, 0] = 1.0
    sim.grid.u[1, 0, 0] = 1.0  # flux from cell 0 -> 1 along x
    sim.grid.pr[1, 0, 0] = -10.0

    mass_before = sim.total_mass()
    sim._flux_to_particles(0.1)
    mass_after = sim.total_mass()

    assert np.isclose(mass_before, mass_after)
    assert sim.x.shape[0] > 0
    assert sim.phi[0, 0, 0] < 1.0


def test_min_velocity_condenses():
    params = HybridParams(dx=1.0, particle_mass=1.0, v_min=0.5, tau_slow=0.0)
    sim = HybridFluid(shape=(1, 1, 1), n_particles=1, params=params)
    sim.x[0] = [0.5, 0.5, 0.5]
    sim.v[0] = [0.1, 0.0, 0.0]

    mass_before = sim.total_mass()
    sim._update_pausing_particles(0.1)
    mass_after = sim.total_mass()

    assert sim.x.shape[0] == 0
    assert np.isclose(mass_before, mass_after)
    assert sim.phi[0, 0, 0] > 0.0

