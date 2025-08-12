import pytest
from sympy import Integer

from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.cell_consts import Cell


def test_saline_api_equilibrium_fracs_sum_to_one():
    cells = [Cell(stride=1, left=0, right=4, label='A'),
             Cell(stride=1, left=4, right=8, label='B')]
    sim = Simulator(cells)
    # seed legacy-style expressions
    sim.s_exprs = [Integer(1), Integer(1)]
    sim.p_exprs = [Integer(1), Integer(1)]

    sim.run_saline_sim()
    fracs = sim.equilibrium_fracs(0.0)
    assert abs(sum(fracs) - 1.0) < 1e-6
