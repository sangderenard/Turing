import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams


def test_lower_bulk_modulus_allows_larger_dt():
    """Lowering bulk modulus should yield a larger stable timestep."""
    positions = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)

    default_params = FluidParams()
    fluid_default = DiscreteFluid(positions, velocities=None, temperature=None,
                                  salinity=None, params=default_params)
    fluid_default._build_grid()
    dt_default = fluid_default._stable_dt()

    soft_params = FluidParams(bulk_modulus=default_params.bulk_modulus * 1e-4)
    fluid_soft = DiscreteFluid(positions, velocities=None, temperature=None,
                               salinity=None, params=soft_params)
    fluid_soft._build_grid()
    dt_soft = fluid_soft._stable_dt()

    assert dt_soft > dt_default * 10

    # Evolve the soft fluid near its stability limit to ensure integration remains finite
    fluid_soft.step(dt_soft * 0.9)
    assert np.all(np.isfinite(fluid_soft.p))
    assert np.all(np.isfinite(fluid_soft.v))
