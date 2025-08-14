from src.cells.cellsim.placement.bitbuffer import BitBufferAdapter
from src.bitbitbuffer import BitBitBuffer, CellProposal
from src.cells.cell_consts import Cell as LegacyCell


def make_cell(label, left, right, stride):
    return LegacyCell(stride, left, right, leftmost=left, rightmost=right-1, label=label)


def test_expand_mapping_lcm_alignment():
    cells = [make_cell("A", 0, 8, 4), make_cell("B", 8, 16, 4)]
    buf = BitBitBuffer(mask_size=16)
    adapter = BitBufferAdapter(buf)
    props = [CellProposal(c) for c in cells]
    props = adapter.expand([5,0], cells, props)
    # expansion rounded to 8 (LCM 4)
    assert props[0].right - props[0].left == 16
    # second cell shifted by 8
    assert props[1].left == 16
    assert buf.mask_size >= 24
