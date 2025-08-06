from sympy import Integer
from .cell_consts import CELL_COUNT
from .salinepressure import SalineHydraulicSystem

def update_s_p_expressions(sim, cells):
    sim.s_exprs = [Integer(cell.salinity) for cell in cells]
    sim.p_exprs = [Integer(cell.pressure) for cell in cells]

def run_saline_sim(self):
    # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
    self.update_s_p_expressions(self.cells)
    self.engine = SalineHydraulicSystem(
        self.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
        self.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
        width=self.bitbuffer.mask_size, # the total bit‐space you’re dividing
        chars=[chr(97+i) for i in range(CELL_COUNT)],
        tau=5, math_type='int',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )
    for cell in self.cells:
        if cell.leftmost is None:
            print(f"Line 67: Cell {cell.label} leftmost is None, setting to left {cell.left}")
            cell.leftmost = cell.left
        if cell.rightmost is None:
            print(f"Line 70: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
            cell.rightmost = cell.right - 1
    # 2) Ask for the equilibrium fractions at t=0
    self.fractions = self.engine.equilibrium_fracs(0.0)
    #for cell in self.cells:
        #if cell.salinity == 0:
            #cell.salinity = 1

    necessary_size = self.bitbuffer.intceil(sum(cell.salinity for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0), self.system_lcm)
        
    if self.bitbuffer.mask_size < necessary_size:
        offsets = [self.bitbuffer.intceil((cell.rightmost - cell.leftmost)//2+cell.leftmost, cell.stride) for cell in self.cells if hasattr(cell, 'leftmost')]
        sizes = [(cell.salinity) for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0]
        size_and_offsets = sorted(list(zip(sizes, offsets)), reverse=True, key=lambda x: x[1])
        for size, offset in size_and_offsets:
            self.expand([offset], self.bitbuffer.intceil(size, self.lcm(self.cells)), self.cells, self.cells)

    self.snap_cell_walls(self.cells, self.cells)
