import math
from cellsim.data.state import Cell, Bath, Organelle
from cellsim.core.geometry import sphere_area_from_volume
from cellsim.engine.saline import SalineEngine

species = ("Na","K","Cl","Imp")

# One cell with one organelle
cell = Cell(V=1.0, n={"Imp":1500.0,"Na":10.0,"K":140.0,"Cl":10.0})
A0, _ = sphere_area_from_volume(cell.V); cell.A0 = A0
cell.organelles = [
    Organelle(volume_total=0.2, lumen_fraction=0.6, n={"Imp":50.0,"Na":2.0,"K":10.0,"Cl":2.0})
]

bath = Bath(V=1.0, n={"Na":1500.0,"K":0.0,"Cl":1500.0,"Imp":0.0}, pressure=1e4, temperature=298.15)

eng = SalineEngine([cell], bath, species=species)
dt = 1e-3
for k in range(1000):
    dt = eng.step(dt)

print("Cell V:", cell.V, "Bath V:", bath.V)
print("Cell Na:", cell.n["Na"], "Bath Na:", bath.n["Na"])
