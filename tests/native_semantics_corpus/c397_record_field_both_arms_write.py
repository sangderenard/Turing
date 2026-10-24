# Both arms write the field; the read is the merge of the two writes.
from dataclasses import dataclass


@dataclass
class S:
    n: int


def root(c: int) -> int:
    s = S(0)
    if c > 0:
        s.n = c * 2
    else:
        s.n = 7 - c
    return s.n
# INPUTS [{'c': 0}, {'c': 3}, {'c': -2}]
