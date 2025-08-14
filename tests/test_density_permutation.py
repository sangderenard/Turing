import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_density_invariance_under_permutation():
    rng = np.random.default_rng(0)
    n = 10
    positions = rng.uniform(-0.2, 0.2, size=(n, 3))
    params = FluidParams()

    fluid = DiscreteFluid(positions, velocities=None, temperature=None, salinity=None, params=params)
    fluid._build_grid()
    fluid._compute_density(include_self=True)
    rho_ref = fluid.rho.copy()

    perm = rng.permutation(n)
    positions_perm = positions[perm]
    fluid_perm = DiscreteFluid(positions_perm, velocities=None, temperature=None, salinity=None, params=params)
    fluid_perm._build_grid()
    fluid_perm._compute_density(include_self=True)
    rho_perm = fluid_perm.rho.copy()

    # reorder permuted densities back to original particle order
    rho_perm_back = rho_perm[np.argsort(perm)]
    assert np.allclose(rho_ref, rho_perm_back)
