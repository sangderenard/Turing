# Use a relative import to access Cell definition from cells package.
from ...cells.cell_consts import Cell

class CellProposal(Cell):
    def __init__(self, cell):

        super().__init__(cell.stride, cell.left, cell.right, cell.len, leftmost=cell.leftmost, rightmost=cell.rightmost, label=cell.label)

        self.salinity = cell.salinity
        self.pressure = cell.pressure
        self.leftmost = cell.leftmost
        self.rightmost = cell.rightmost
