# The constructor argument is a parameter; a later conditional write replaces
# it, and the read sits inside an expression.
from dataclasses import dataclass


@dataclass
class S:
    n: int
    m: int


def root(c: int) -> int:
    s = S(c, 4)
    if c > 1:
        s.n = 1
    return s.n * 10 + s.m
# INPUTS [{'c': 0}, {'c': 1}, {'c': 5}]
