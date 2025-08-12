from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict

@dataclass
class GateState:
    name: str
    g_open: float = 0.0  # conductance or permeability surrogate

@dataclass
class GateModel:
    states: List[GateState]
    edges: List[Tuple[int,int,Callable[...,float]]]  # rate(i->j; Vm, lig, tension)->k
    selectivity: Dict[str, float]
    flux_law: str  # "GHK"|"Pore"|"Carrier"

    def step_markov(self, Vm: float, lig: dict, tension: float, dt: float, rng) -> None:
        # Stub: hook for SSA/CTMC update
        return

    def open_fraction(self) -> float:
        # Stub: compute occupancy of "open" states
        return 0.0
