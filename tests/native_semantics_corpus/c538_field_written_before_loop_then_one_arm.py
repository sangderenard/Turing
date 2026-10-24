# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
    m: int
def root(a: int, k: int) -> int:
    s = S(a, 0)
    s.n = s.n + 5
    for i in range(k):
        if s.n % 2 == 0:
            s.n = s.n // 2
        s.m += 1
    return s.n * 100 + s.m
# INPUTS [{'a': 3, 'k': 4}, {'a': -2, 'k': 0}, {'a': 11, 'k': 3}]
