# A one-field record whose field folds to the argument it is merged with:
# the argument read after the merge is the argument (silently the merge on main).
from dataclasses import dataclass
@dataclass
class R:
    x: int
def root(n: int, c: int) -> int:
    r = R(n)
    if c > 2:
        r.x = n + 100
    return r.x * 10 + n
# INPUTS [{'n': 3, 'c': 5}, {'n': -7, 'c': 2}]
