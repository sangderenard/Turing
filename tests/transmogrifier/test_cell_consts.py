from transmogrifier.cells.cell_consts import Cell, DEFAULT_FLAG_PROFILES


def test_cell_flag_profiles():
    cell = Cell(stride=1, left=0, right=4, len=4, profile="default")
    flags = DEFAULT_FLAG_PROFILES["default"]
    assert cell.l_wall_flags == flags["left_wall"]
    assert cell.r_wall_flags == flags["right_wall"]
    assert cell.c_flags == flags["cell"]
    assert cell.system_flags == flags["system"]
