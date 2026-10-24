# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
def root(a: int, k: int) -> int:
    s = S(a)
    for i in range(k):
        s.n = s.n * 2
        if s.n > 50:
            return s.n + i
    return -s.n
# INPUTS [{'a': 3, 'k': 10}, {'a': -2, 'k': 3}, {'a': 1, 'k': 0}]
