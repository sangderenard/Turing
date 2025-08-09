from sympy import Integer
from src.transmogrifier.cells.simulator_methods.salinepressure import SalineHydraulicSystem


def test_equilibrium_fracs_sum_to_one():
    system = SalineHydraulicSystem([Integer(1), Integer(1)], [Integer(1), Integer(1)], width=10)
    fracs = system.equilibrium_fracs(0.0)
    assert abs(sum(fracs) - 1.0) < 1e-6
