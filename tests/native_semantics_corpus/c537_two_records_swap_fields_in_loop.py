# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
    m: int
def root(a: int, k: int) -> int:
    s = S(a, 0)
    t = S(0, a)
    for i in range(k):
        s.n, t.n = t.n + 1, s.n
        t.m = t.m - i
    return s.n * 10000 + t.n * 100 + t.m
# INPUTS [{'a': 3, 'k': 4}, {'a': -2, 'k': 0}, {'a': 7, 'k': 1}]
