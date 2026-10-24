# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
    m: int
def root(a: int) -> int:
    s = S(a, 1)
    for i in range(3):
        s.m = s.m * 2 + s.n
    return s.m
# INPUTS [{'a': 3}, {'a': -2}]
