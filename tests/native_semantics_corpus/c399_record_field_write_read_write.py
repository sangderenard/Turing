# Unconditional write, read, conditional write, read: each read sees the
# version current at its own place.
from dataclasses import dataclass


@dataclass
class S:
    n: int


def root(c: int) -> int:
    s = S(1)
    s.n = c + 2
    first = s.n
    if c > 0:
        s.n = first * 3
    return first + s.n
# INPUTS [{'c': 0}, {'c': 2}, {'c': -5}]
