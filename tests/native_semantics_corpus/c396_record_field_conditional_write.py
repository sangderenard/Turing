# A field written inside an if must be read back after it, not folded to the
# constructor argument (silent WRONG at ef49715: c=3 returned 0).
from dataclasses import dataclass


@dataclass
class S:
    n: int


def root(c: int) -> int:
    s = S(0)
    if c > 0:
        s.n = 1
    return s.n
# INPUTS [{'c': 0}, {'c': 3}, {'c': -2}]
