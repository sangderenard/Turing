from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Reaction:
    """Mass-action reaction ν_reactants → ν_products with rate constant k."""
    nu_react: Dict[str, int]
    nu_prod: Dict[str, int]
    k: float

    def rate(self, conc: Dict[str, float]) -> float:
        v = self.k
        for sp, sto in self.nu_react.items():
            v *= conc.get(sp, 0.0) ** sto
        return v

@dataclass
class CRN:
    species: List[str]
    reactions: List[Reaction] = field(default_factory=list)

    def v(self, conc: Dict[str,float]) -> Dict[str,float]:
        prod = {sp: 0.0 for sp in self.species}
        for rxn in self.reactions:
            r = rxn.rate(conc)
            for sp in self.species:
                nu = rxn.nu_prod.get(sp,0) - rxn.nu_react.get(sp,0)
                if nu != 0:
                    prod[sp] += nu * r
        return prod
