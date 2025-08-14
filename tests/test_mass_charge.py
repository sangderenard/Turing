import math
from src.cells.cellsim.engine.saline import SalineEngine
from src.cells.cellsim.data.state import Cell, Bath
from src.cells.cellsim.core.geometry import sphere_area_from_volume


def test_mass_conserved_without_pump():
    cell = Cell(V=10.0, n={"Na":10.0, "K":5.0})
    bath = Bath(V=100.0, n={"Na":100.0, "K":50.0})
    eng = SalineEngine([cell], bath, species=("Na","K"))
    before = {sp: cell.n[sp] + bath.n[sp] for sp in ("Na","K")}
    eng.step(1.0)
    after = {sp: cell.n[sp] + bath.n[sp] for sp in ("Na","K")}
    assert math.isclose(after["Na"], before["Na"], rel_tol=1e-6)
    assert math.isclose(after["K"], before["K"], rel_tol=1e-6)


def test_pump_stoichiometry():
    cell = Cell(V=10.0, n={"Na":10.0, "K":5.0})
    bath = Bath(V=100.0, n={"Na":100.0, "K":50.0})
    rate = 1e-3
    eng = SalineEngine([cell], bath, species=("Na","K"))
    area, _ = sphere_area_from_volume(cell.V)
    cell.J_pump = rate * area
    eng.step(1.0)
    expected = cell.J_pump * 1.0
    assert math.isclose(cell.n["Na"], 10.0 - 3*expected, rel_tol=1e-6)
    assert math.isclose(bath.n["Na"], 100.0 + 3*expected, rel_tol=1e-6)
    assert math.isclose(cell.n["K"], 5.0 + 2*expected, rel_tol=1e-6)
    assert math.isclose(bath.n["K"], 50.0 - 2*expected, rel_tol=1e-6)
