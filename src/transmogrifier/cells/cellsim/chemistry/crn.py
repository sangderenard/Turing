from dataclasses import dataclass, field
from typing import Dict, List, Callable
import numpy as np

@dataclass
class Reaction:
    # ν_reactants -> ν_products
    nu_react: Dict[str, int]
    nu_prod: Dict[str, int]
    rate_law: Callable[[Dict[str,float]], float]  # v = f(conc)

@dataclass
class CRN:
    species: List[str]
    reactions: List[Reaction] = field(default_factory=list)

    def v(self, conc: Dict[str,float]) -> Dict[str,float]:
        # total production rate by species (placeholder)
        return {sp: 0.0 for sp in self.species}
