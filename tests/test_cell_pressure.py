import random
import os
import pytest
from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.cell_consts import Cell
from src.transmogrifier.cells.bitbitbuffer import BitBitBuffer

def test_simulation_stride_basic(stride):
    random.seed(0)
    CELL_COUNT = random.randint(1, 5)
    WIDTH      = stride * 8
    cells = [Cell(stride=stride,
                  left=i * WIDTH,
                  len=WIDTH,
                  right=i * WIDTH + WIDTH)
             for i in range(CELL_COUNT)]
    sim = Simulator(cells)
    sp, _ = sim.step(cells)
    assert isinstance(sp, (int, float))
    for c in cells:
        assert len(sim.get_cell_mask(c)) == c.right - c.left

def test_injection_mixed_prime7():
    stride = 7
    CELL_COUNT = 3
    WIDTH = stride * 20
    cells = [Cell(stride=stride,
                  left=i * WIDTH,
                  len=WIDTH,
                  right=i * WIDTH + WIDTH,
                  label=f"cell{i}")
             for i in range(CELL_COUNT)]
    sim = Simulator(cells)
    data_bytes_per_stride = (stride * sim.bitbuffer.bitsforbits + 7) // 8
    payloads = [
        b'\xff' * data_bytes_per_stride,
        b'\xaa' * data_bytes_per_stride,
        b'\x55' * data_bytes_per_stride
    ]
    for p in payloads:
        sim.write_data(cells[0].label, p)
    for _ in range(10):
        sp, _ = sim.step(cells)
    sim.print_system()
    assert cells[0].injection_queue == 0

def test_sustained_random_injection():
    CELL_STRIDES = [7, 11, 13, 17]
    CELL_COUNT = len(CELL_STRIDES)
    INITIAL_TARGET = 300
    SIMULATION_STEPS = 50
    WRITES_PER_STEP = 50
    cells = [
        Cell(
            stride=s,
            left=i * BitBitBuffer._intceil(INITIAL_TARGET, s),
            len=BitBitBuffer._intceil(INITIAL_TARGET, s),
            right=(i + 1) * BitBitBuffer._intceil(INITIAL_TARGET, s),
            label=f"cell_{s}",
        )
        for i, s in enumerate(CELL_STRIDES)
    ]
    sim = Simulator(cells)
    for step in range(SIMULATION_STEPS):
        for _ in range(random.randint(1, WRITES_PER_STEP)):
            target_cell = random.choice(cells)
            data_bytes = (target_cell.stride * sim.bitbuffer.bitsforbits + 7) // 8
            payload = os.urandom(data_bytes)
            sim.write_data(target_cell.label, payload)
        sim.step(cells)
    total_remaining_items = 0
    for cell in cells:
        assert cell.injection_queue == 0
        remaining_in_queue = len(sim.input_queues.get(cell.label, []))
        assert remaining_in_queue == 0
        total_remaining_items += remaining_in_queue
    assert total_remaining_items == 0
