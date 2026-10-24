# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
    m: int
def root(a: int, k: int) -> int:
    s = S(a, 0)
    for i in range(k):
        if i % 2 == 0:
            s.n = s.n * 2
        else:
            s.m = s.m + s.n
    return s.n * 1000 + s.m
# INPUTS [{'a': 3, 'k': 5}, {'a': -2, 'k': 0}, {'a': 5, 'k': 2}]
