from sympy import Integer
from ..cell_consts import CELL_COUNT
from ..salinepressure import SalineHydraulicSystem

def update_s_p_expressions(sim, cells):
    sim.s_exprs = [Integer(cell.salinity) for cell in cells]
    sim.p_exprs = [Integer(cell.pressure) for cell in cells]

def run_saline_sim(sim):
    update_s_p_expressions(sim, sim.cells)
    sim.engine = SalineHydraulicSystem(
        sim.s_exprs, sim.p_exprs, width=sim.bitbuffer.mask_size,
        chars=[chr(97+i) for i in range(CELL_COUNT)],
        tau=5, math_type='int',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )
    sim.fractions = sim.engine.equilibrium_fracs(0.0)
