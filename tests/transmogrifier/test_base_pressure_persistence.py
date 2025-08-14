import pytest

from src.cells.cell_consts import Cell
from src.cells.simulator import Simulator
from src.cells.cellsim.api.saline import SalinePressureAPI


def test_base_pressure_preserved_from_legacy():
    cells = [Cell(stride=1, left=0, right=8, label='A')]
    cells[0].base_pressure = 2.5
    cells[0].pressure = 3.5
    sim = Simulator(cells)

    api = SalinePressureAPI.from_legacy(sim)

    assert api.cells[0].base_pressure == pytest.approx(2.5)
