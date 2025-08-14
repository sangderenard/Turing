from dataclasses import dataclass

@dataclass(frozen=True)
class Species:
    name: str
    z: int = 0
    D: float = 1e-9              # m^2/s
    radius: float | None = None  # m
    Kp_membrane: float = 1.0     # unitless partition coefficient
    activity_model: str = "ideal"

class SpeciesRegistry(dict):
    def add(self, sp: Species):
        self[sp.name] = sp
