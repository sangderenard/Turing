# A field of a record this function constructs, written in a loop: a carried
# binding like a rebound name (refused as loop_exit_field_state_unmerged before).
from dataclasses import dataclass
@dataclass
class S:
    n: int
def root(a: int) -> int:
    s = S(a)
    c = 0
    while s.n > 1:
        s.n = s.n // 2
        c += 1
    return c * 100 + s.n
# INPUTS [{'a': 37}, {'a': 1}, {'a': 2}]
