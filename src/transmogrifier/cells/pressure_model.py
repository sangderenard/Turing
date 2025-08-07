from sympy import Integer, Float
from .cell_consts import CELL_COUNT
from .salinepressure import SalineHydraulicSystem

def balance_system(cells, mode='open'):
    """Balance salinity and pressure across the whole system."""
    if not cells or mode not in ('open', 'closed'):
        return
    if mode == 'open':
        total_s = sum(c.salinity for c in cells) or 1
        total_p = sum(c.pressure for c in cells) or 1
        for c in cells:
            c.salinity = c.salinity / total_s
            c.pressure = c.pressure / total_p
    else:  # closed system – zero-sum
        avg_s = sum(c.salinity for c in cells) / len(cells)
        avg_p = sum(c.pressure for c in cells) / len(cells)
        for c in cells:
            c.salinity -= avg_s
            c.pressure -= avg_p

def update_s_p_expressions(sim, cells, *, as_float=False):
    if as_float:
        sim.s_exprs = [Float(cell.salinity) for cell in cells]
        sim.p_exprs = [Float(cell.pressure) for cell in cells]
    else:
        sim.s_exprs = [Integer(cell.salinity) for cell in cells]
        sim.p_exprs = [Integer(cell.pressure) for cell in cells]
def equilibrium_fracs(sim, t):
    if not hasattr(sim, 'engine') or sim.engine is None:
        run_saline_sim(sim)
    return sim.engine.equilibrium_fracs(t)
def run_saline_sim(sim, *, as_float=False):
    # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
    update_s_p_expressions(sim, sim.cells, as_float=as_float)
    sim.engine = SalineHydraulicSystem(
        sim.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
        sim.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
        width=sim.bitbuffer.mask_size, # the total bit‐space you’re dividing
        chars=[chr(97+i) for i in range(CELL_COUNT)],
        tau=5, math_type='int',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )
    for cell in sim.cells:
        if cell.leftmost is None:
            #print(f"Line 67: Cell {cell.label} leftmost is None, setting to left {cell.left}")
            cell.leftmost = cell.left
        if cell.rightmost is None:
            #print(f"Line 70: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
            cell.rightmost = cell.right - 1
    # 2) Ask for the equilibrium fractions at t=0
    sim.fractions = sim.engine.equilibrium_fracs(0.0)
    #for cell in sim.cells:
        #if cell.salinity == 0:
            #cell.salinity = 1

    necessary_size = sim.bitbuffer.intceil(sum(cell.salinity for cell in sim.cells if hasattr(cell, 'salinity') and cell.salinity > 0), sim.system_lcm)

    if sim.bitbuffer.mask_size < necessary_size:
        offsets = [sim.bitbuffer.intceil((cell.rightmost - cell.leftmost)//2+cell.leftmost, cell.stride) for cell in sim.cells if hasattr(cell, 'leftmost')]
        sizes = [(cell.salinity) for cell in sim.cells if hasattr(cell, 'salinity') and cell.salinity > 0]
        size_and_offsets = sorted(list(zip(sizes, offsets)), reverse=True, key=lambda x: x[1])
        for size, offset in size_and_offsets:
            sim.expand([offset], sim.bitbuffer.intceil(size, sim.lcm(sim.cells)), sim.cells, sim.cells)

    sim.snap_cell_walls(sim.cells, sim.cells)


def run_balanced_saline_sim(sim, mode='open'):
    """Balance the system then run the standard saline simulation."""
    balance_system(sim.cells, mode)
    run_saline_sim(sim, as_float=True)
