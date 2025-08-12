from dataclasses import dataclass, field
from typing import Dict, List
from ..core.geometry import sphere_area_from_volume

@dataclass
class Compartment:
    V: float                              # volume (m^3-ish)
    phi: float = 0.0                      # electric potential (V), future
    n: Dict[str, float] = field(default_factory=dict)  # moles by species

    def conc(self, species: List[str]) -> dict:
        V = max(self.V, 1e-18)
        return {sp: self.n.get(sp, 0.0)/V for sp in species}

@dataclass
class Organelle:
    volume_total: float
    lumen_fraction: float = 0.7
    n: Dict[str, float] = field(default_factory=dict)
    Lp0: float = 0.01
    Ps0: Dict[str, float] = field(default_factory=lambda: {"Na":0.01,"K":0.01,"Cl":0.01,"Imp":0.0})
    sigma: Dict[str, float] = field(default_factory=lambda: {"Na":0.9,"K":0.9,"Cl":0.9,"Imp":1.0})
    Ea_Lp: float | None = None
    Ea_Ps: Dict[str, float] = field(default_factory=lambda: {"Na":None,"K":None,"Cl":None,"Imp":None})
    anchor_stiffness: float = float("inf")
    eps_ref: float = 0.0

    def V_lumen(self) -> float:
        return max(self.volume_total * self.lumen_fraction, 1e-18)

    # Provide a Compartment-like interface for transport code
    @property
    def V(self) -> float:
        return self.V_lumen()

    def conc(self, species: list[str]) -> dict:
        V = max(self.V, 1e-18)
        return {sp: self.n.get(sp, 0.0) / V for sp in species}

@dataclass
class Cell(Compartment):
    A0: float = 0.0
    elastic_k: float = 0.1
    visc_eta: float = 0.0
    Lp0: float = 1.0
    Ps0: Dict[str, float] = field(default_factory=lambda: {"Na":0.01,"K":0.01,"Cl":0.01,"Imp":0.0})
    sigma: Dict[str, float] = field(default_factory=lambda: {"Na":0.9,"K":0.9,"Cl":0.9,"Imp":1.0})
    Ea_Lp: float | None = None
    Ea_Ps: Dict[str, float] = field(default_factory=lambda: {"Na":None,"K":None,"Cl":None,"Imp":None})
    organelles: List[Organelle] = field(default_factory=list)
    base_pressure: float = 1e4
    _prev_eps: float = 0.0  # runtime
    # Na/K pump controls
    pump_enabled: bool = False
    pump_Jmax: float = 0.0
    pump_Km_Nai: float = 10.0
    pump_Km_Ko: float = 1.5
    pump_alpha_tension: float = 0.0
    J_pump: float = 0.0

    def set_initial_A0_if_missing(self):
        if self.A0 <= 0.0:
            A0, _ = sphere_area_from_volume(self.V)
            self.A0 = A0

@dataclass
class Bath(Compartment):
    pressure: float = 1e4
    temperature: float = 298.15
    compressibility: float = 0.0
