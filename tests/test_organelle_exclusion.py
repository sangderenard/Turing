from src.transmogrifier.cells.cellsim.data.state import Cell, Bath, Organelle
from src.transmogrifier.cells.cellsim.engine.saline import SalineEngine
from src.transmogrifier.cells.cellsim.organelles.inner_loop import cytosol_free_volume


def test_organelle_exclusion():
    c1 = Cell(V=10.0, n={"Imp":10.0})
    c1.organelles.append(Organelle(volume_total=2.0, lumen_fraction=0.0, n={"Imp":0.0}))
    c2 = Cell(V=10.0, n={"Imp":10.0})
    bath = Bath(V=100.0, n={"Imp":0.0})
    SalineEngine([c1, c2], bath, species=("Imp",))
    V1 = cytosol_free_volume(c1)
    V2 = cytosol_free_volume(c2)
    assert V1 < V2
    assert c1.n["Imp"] / V1 > c2.n["Imp"] / V2
