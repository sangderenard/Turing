from transmogrifier.cells.cell_consts import Cell
from transmogrifier.cells.bitbitbuffer import BitBitBuffer
from transmogrifier.cells.cell_pressure_region_manager import CellPressureRegionManager


def test_quanta_map_reports_usage():
    c1 = Cell(stride=1, left=0, right=8, len=8, label="A")
    c2 = Cell(stride=1, left=8, right=16, len=8, label="B")
    bb = BitBitBuffer(mask_size=16, bitsforbits=8, make_pid=False)
    bb[0] = 1
    bb[8] = 1

    mgr = CellPressureRegionManager(bb, [c1, c2])
    info = mgr.quanta_map()
    assert info[0]["used"] == [(0, 1)]
    assert info[1]["used"] == [(8, 1)]
