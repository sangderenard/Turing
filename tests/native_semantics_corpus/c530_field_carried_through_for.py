# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
def root(a: int, k: int) -> int:
    s = S(a)
    t = 0
    for i in range(k):
        t += a
        s.n = s.n + i
    return s.n * 100 + t
# INPUTS [{'a': 3, 'k': 4}, {'a': -2, 'k': 0}, {'a': 5, 'k': 1}]
