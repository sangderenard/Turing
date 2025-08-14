from src.cells.cell_consts import Cell
from src.cells.simulator import Simulator
from src.bitbitbuffer import CellProposal


def test_snap_cell_walls_defers_cell_sync_until_after_expand():
    """Expanding past the mask should not raise and cells update afterward."""
    cells = [
        Cell(label="A", left=0, right=8, stride=8, leftmost=0, rightmost=7),
        Cell(label="B", left=8, right=16, stride=8, leftmost=8, rightmost=15),
    ]
    sim = Simulator(cells)

    proposals = [CellProposal(c) for c in cells]
    proposals[1].right = 40  # push beyond current mask

    sim.snap_cell_walls(cells, proposals)

    assert cells[1].right >= 40
    assert sim.bitbuffer.mask_size > 16
