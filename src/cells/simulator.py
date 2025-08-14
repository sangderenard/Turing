from typing import Union
from sympy import Integer
from .cell_consts import Cell, MASK_BITS_TO_DATA_BITS, CELL_COUNT
# Prefer the new cellsim API; fall back to legacy if needed

from .cellsim.api.saline import (
    run_saline_sim as cs_run_saline_sim,
    balance_system as cs_balance_system,
    update_s_p_expressions as cs_update_s_p_expressions,
    equilibrium_fracs as cs_equilibrium_fracs,
    run_balanced_saline_sim as cs_run_balanced,
    )

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Only for type hints; avoid import-time circular deps
    from ..bitbitbuffer import BitBitBuffer as _BBB_Type, CellProposal as _CP_Type
    from ..bitbitbuffer.helpers.bitstream_search import BitStreamSearch as _BSS_Type
from .cell_walls import snap_cell_walls, build_metadata, expand
import math
import random
import os


class Simulator:
    FORCE_THRESH = .5
    LOCK = 0x1
    ELASTIC = 0x2
    SALINE_BUFFER = 0.1  # fraction of capacity reserved before expansion
    snap_cell_walls = snap_cell_walls
    build_metadata = build_metadata
    expand = expand

    def __init__(self, cells):
        self.assignable_gaps = {}
        self.pid_list = []
        self.cells = cells
        self.input_queues = {}
        self.system_pressure = 0
        self.elastic_coeff = 0.1
        self.epsilon = 1e-6
        self.N = len(cells)
        self.system_lcm   = self.lcm(cells)
        required_end = max(c.right for c in cells)
        # Lazy import here to avoid circular import during package init
        from ..bitbitbuffer import BitBitBuffer
        from ..bitbitbuffer.helpers.bitstream_search import BitStreamSearch
        mask_size    = BitBitBuffer._intceil(required_end, self.system_lcm)
        self.bitbuffer = BitBitBuffer(
            mask_size=mask_size,
            caster=bytes,
            bitsforbits=MASK_BITS_TO_DATA_BITS,
        )
        self.bitbuffer.register_pid_buffer(cells=self.cells)
        self.locked_data_regions = []
        self.search = BitStreamSearch()
        # Initial sympy expressions for legacy view
        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]
        # Engine handles and state
        self.engine = None
        self.fractions = None
        self.closed = False

# Attach modularized methods from simulator_methods
from .simulator_methods.visualization import print_system, bar, crosscheck
from .simulator_methods.cell_mask import get_cell_mask, set_cell_mask, pull_cell_mask, push_cell_mask
from .simulator_methods.data_io import actual_data_hook

from .simulator_methods.injection import injection
from .simulator_methods.quanta_map_and_dump_cells import quanta_map, dump_cells
from .simulator_methods.lcm import lcm
from .simulator_methods.minimize import minimize

Simulator.print_system = print_system
Simulator.bar = staticmethod(bar)
Simulator.get_cell_mask = get_cell_mask
Simulator.set_cell_mask = set_cell_mask
Simulator.pull_cell_mask = pull_cell_mask
Simulator.push_cell_mask = push_cell_mask
#this was an inapropriate bypass
#Simulator.write_data = write_data
#Simulator.flush_pending_writes = flush_pending_writes
Simulator.actual_data_hook = actual_data_hook

Simulator.injection = injection
Simulator.quanta_map = quanta_map
Simulator.dump_cells = dump_cells
Simulator.lcm = lcm
Simulator.minimize = minimize
Simulator.crosscheck = crosscheck
Simulator.run_saline_sim = cs_run_saline_sim
Simulator.run_balanced_saline_sim = cs_run_balanced
Simulator.update_s_p_expressions = cs_update_s_p_expressions
Simulator.equilibrium_fracs = cs_equilibrium_fracs
Simulator.balance_system = cs_balance_system


# Optional: apply relocation preprocessing plan (stride-aligned moves and contractions)
def _apply_relocation_preprocessing(self, object_indices_to_move, destination_indices):
    """THIS IS NOT FOR EXPAND, THIS IS MOVING STRIDE FOR STRIDE EXCHANGES"""
    self.bitbuffer.relocate(object_indices_to_move, destination_indices)

Simulator.apply_relocation_preprocessing = _apply_relocation_preprocessing




def main():
    """
    A simple demonstration of the cell simulation.
    This function is intended to be run as a script.
    """
    from .cell_consts import Cell
    cells = [
        Cell(label='A', left=0, right=16, stride=4),
        Cell(label='B', left=16, right=32, stride=4),
    ]
    sim = Simulator(cells)

    for _ in range(5):
        
        sim.input_queues['A'] = [(b'\xde\xad' * 4, 4)]
        sim.input_queues['B'] = [(b'\xbe\xef' * 4, 4)]
        cells[0].injection_queue = 1
        cells[1].injection_queue = 1
        sim.minimize(sim.cells)
        sim.print_system()

if __name__ == "__main__":
    main()
