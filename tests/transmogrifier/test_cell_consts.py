from src.cells.cell_consts import Cell, DEFAULT_FLAG_PROFILES
from src.cells.cellsim.constants import (
    DEFAULT_ELASTIC_K,
    DEFAULT_LP0,
    SALINITY_PER_DATA_UNIT,
)
from src.cells.cellsim.data.state import Organelle


def test_cell_flag_profiles():
    cell = Cell(stride=1, left=0, right=4, len=4, profile="default")
    flags = DEFAULT_FLAG_PROFILES["default"]
    assert cell.l_wall_flags == flags["left_wall"]
    assert cell.r_wall_flags == flags["right_wall"]
    assert cell.c_flags == flags["cell"]
    assert cell.system_flags == flags["system"]


def test_cell_biophysical_defaults():
    c = Cell(stride=1, left=0, right=4, len=4)
    assert c.elastic_coeff == DEFAULT_ELASTIC_K
    assert c.l_solvent_permiability == DEFAULT_LP0
    assert c.salinity_per_data_unit == SALINITY_PER_DATA_UNIT


def test_organelle_salinity_seed():
    o = Organelle(volume_total=1.0)
    assert o.n["Imp"] == o.V_lumen() * SALINITY_PER_DATA_UNIT
