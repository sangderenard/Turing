# A field merge whose pre-branch value folds to a constructor argument: the
# argument must keep its own value after the merge (silently returned the
# merge for later reads of n before; fuzz seed 99 p99_24).
from dataclasses import dataclass
@dataclass
class Rec:
    x: int
    y: int
def root(n: int) -> int:
    r1 = Rec(n, 3)
    if n > 7:
        r1.x = n + 6
    return r1.x * 10 + n
# INPUTS [{'n': 9}, {'n': 12}, {'n': 3}]
