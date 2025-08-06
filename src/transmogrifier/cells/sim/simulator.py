from typing import Union
from sympy import Integer
from ..cell_consts import Cell, MASK_BITS_TO_DATA_BITS, CELL_COUNT, RIGHT_WALL, LEFT_WALL
from ..salinepressure import SalineHydraulicSystem
from ..bitbitbuffer import BitBitBuffer, CellProposal
from ..bitstream_search import BitStreamSearch
import math
import random
import os

class Simulator:
    FORCE_THRESH = .5
    LOCK = 0x1
    ELASTIC = 0x2

    def __init__(self, cells):
        self.assignable_gaps = {}
        self.pid_list = []
        self.cells = cells
        self.input_queues = {}
        self.system_pressure = 0
        self.elastic_coeff = 0.1
        self.system_lcm   = self.lcm(cells)
        required_end = max(c.right for c in cells)
        mask_size    = BitBitBuffer._intceil(required_end, self.system_lcm)
        self.bitbuffer = BitBitBuffer(mask_size=mask_size, caster=bytes,
                                    bitsforbits=MASK_BITS_TO_DATA_BITS)
        self.bitbuffer.register_pid_buffer(cells=self.cells)
        self.locked_data_regions = []
        self.search = BitStreamSearch()
        self.s_exprs = [Integer(0) for _ in range(CELL_COUNT)]
        self.p_exprs = [Integer(1) for _ in range(CELL_COUNT)]
        self.engine = None
        self.fractions = None
        self.run_saline_sim()

    def update_s_p_expressions(self, cells):
        self.s_exprs = [Integer(cell.salinity) for cell in cells]
        self.p_exprs = [Integer(cell.pressure) for cell in cells]

    def run_saline_sim(self):
        self.update_s_p_expressions(self.cells)
        self.engine = SalineHydraulicSystem(
            self.s_exprs, self.p_exprs, width=self.bitbuffer.mask_size,
            chars=[chr(97+i) for i in range(CELL_COUNT)],
            tau=5, math_type='int',
            int_method='adams',
            protect_under_one=True,
            bump_under_one=True
        )
        self.fractions = self.engine.equilibrium_fracs(0.0)
        necessary_size = self.bitbuffer.intceil(sum(cell.salinity for cell in self.cells if hasattr(cell, 'salinity') and cell.salinity > 0), self.system_lcm)
        if self.bitbuffer.mask_size < necessary_size:
            pass # Expansion logic can be added here
        from .cell_walls import snap_cell_walls
        snap_cell_walls(self, self.cells, self.cells)

    def get_cell_mask(self, cell: Cell) -> bytearray:
        return self.bitbuffer[cell.left:cell.right]

    def set_cell_mask(self, cell: Cell, mask: bytearray) -> None:
        self.bitbuffer[cell.left:cell.right] = mask

    def pull_cell_mask(self, cell):
        cell._buf = self.get_cell_mask(cell)
    def push_cell_mask(self, cell):
        self.set_cell_mask(cell, cell._buf)

    def evolution_tick(self, cells):
        self.engine.s_funcs = [
            (lambda _t, s=cell.salinity: s)
            for cell in cells
        ]
        self.engine.p_funcs = [
            (lambda _t, p=cell.pressure: p)
            for cell in cells
        ]
        proposals = []
        fractions = self.engine.equilibrium_fracs(0.0)
        total_space = self.bitbuffer.mask_size
        for cell, frac in zip(cells, fractions):
            new_width = max(self.bitbuffer.intceil(cell.salinity,cell.stride), self.bitbuffer.intceil(int(total_space * frac), cell.stride))
            assert new_width % cell.stride == 0
            assert cell.stride > 0
            proposal = CellProposal(cell)
            proposals.append(proposal)
        from .cell_walls import snap_cell_walls
        snap_cell_walls(self, cells, proposals)
        from .visualization import print_system
        print_system(self, cells)
        return proposals

    def write_data(self, cell_label: str, payload: bytes):
        try:
            cell = next(c for c in self.cells if c.label == cell_label)
            stride = cell.stride
        except StopIteration:
            raise KeyError(f"No cell with label {cell_label!r}")
        expected_bytes = (stride * self.bitbuffer.bitsforbits + 7) // 8
        if len(payload) != expected_bytes:
            raise ValueError(
                f"Payload for cell '{cell_label}' has incorrect size. "
                f"Expected {expected_bytes} bytes for stride {stride}, but got {len(payload)}."
            )
        self.input_queues.setdefault(cell_label, []).append((payload, stride))
        cell.injection_queue = getattr(cell, "injection_queue", 0) + 1

    def injection(self, queue, known_gaps, gap_pids, left_offset=0):
        consumed_gaps = []
        relative_consumed_gaps = []
        data_copy = queue.copy()
        for i, (payload, stride) in enumerate(data_copy):
            if len(known_gaps) > 0:
                gap = known_gaps.pop()
                if gap >= self.bitbuffer.data_size:
                    exit()
                relative_consumed_gaps.append(gap)
                gap += left_offset
                consumed_gaps.append(gap)
                self.pid_list.append((gap, gap_pids[i]))
                assert stride == len(payload) / self.bitbuffer.bitsforbits * 8
                self.actual_data_hook(payload, gap, stride)
            else:
                break
        return relative_consumed_gaps, consumed_gaps, queue

    def actual_data_hook(self, payload: bytes, dst_bits: int, length_bits: int):
        self.bitbuffer._data_access[dst_bits : dst_bits + length_bits] = payload

    def step(self, cells):
        sp, mask = self.minimize(cells)
        self.evolution_tick(cells)
        return sp, mask

    def minimize(self, cells):
        # ...existing code...
        pass # Move minimize logic here

    def lcm(self, cells):
        from math import gcd
        from functools import reduce
        def lcm(a, b):
            return a * b // gcd(a, b)
        return reduce(lcm, (cell.stride for cell in cells if hasattr(cell, 'stride')), 1)
