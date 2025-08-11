from sympy import Integer
import pytest
from src.transmogrifier.cells.simulator_methods.salinepressure import SalineHydraulicSystem
from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.cell_consts import Cell


def test_equilibrium_fracs_sum_to_one():
    system = SalineHydraulicSystem([Integer(1), Integer(1)], [Integer(1), Integer(1)], width=10)
    fracs = system.equilibrium_fracs(0.0)
    assert abs(sum(fracs) - 1.0) < 1e-6


def test_balance_system_updates_state():
    cells = [Cell(stride=1, left=0, right=4, label='A'),
             Cell(stride=1, left=4, right=8, label='B')]
    sim = Simulator(cells)
    cells[0].salinity = 10.0
    cells[1].salinity = 0.0
    sim.balance_system(cells, sim.bitbuffer, C_ext=3.0, p_ext=0.0, dt=1e-4, max_steps=5)
    assert cells[0].concentration == pytest.approx(cells[0].salinity / cells[0].volume)
    expected = cells[0].base_pressure + cells[0].elastic_coeff * (
        cells[0].volume / cells[0].reference_volume - 1.0
    )
    assert cells[0].pressure == pytest.approx(expected)
