import pytest
from src.transmogrifier.cells.cellsim.api.saline import SalinePressureAPI as SalineHydraulicSystem
from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.cell_consts import Cell
from src.transmogrifier.cells.cellsim.data.state import Cell as SimCell, Bath


def test_equilibrium_fracs_sum_to_one():
    cells = [SimCell(V=1.0, n={"Imp": 1.0}), SimCell(V=1.0, n={"Imp": 1.0})]
    bath = Bath(V=1.0, n={"Imp": 1.0})
    system = SalineHydraulicSystem(cells, bath, width=10)
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

    import math

    def sphere_R(V):
        return (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)

    def sphere_A(V):
        R = sphere_R(V)
        return 4.0 * math.pi * R * R, R

    A_curr, R_curr = sphere_A(cells[0].volume)
    strain = max(A_curr / cells[0].A0 - 1.0, 0.0)
    expected = cells[0].base_pressure + (
        2.0 * (cells[0].elastic_k * strain) / max(R_curr, 1e-12)
    )
    assert cells[0].pressure == pytest.approx(expected)
