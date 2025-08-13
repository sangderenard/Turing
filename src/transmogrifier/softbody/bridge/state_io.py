
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ZeroDCell:
    id: str
    target_volume: float
    osmotic_pressure: float
    membrane_tension: float
    external_pressure: float

@dataclass
class ZeroDState:
    bath_pressure: float
    cells: List[ZeroDCell]

@dataclass
class ZeroDDelta:
    cell_contact_pressures: Dict[str, float]
    cell_surface_areas: Dict[str, float]
    cell_volumes: Dict[str, float]
