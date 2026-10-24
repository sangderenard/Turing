# Field merges inside a loop whose arms read the constructor arguments, with
# the arguments read again after the loop.
from dataclasses import dataclass
@dataclass
class Rec:
    x: int
    y: int
def root(n: int, m: int) -> int:
    r = Rec(n, m)
    t = 0
    for i in range(3):
        if i > m:
            r.x = r.x + i
        else:
            r.y = n * 2
        t = t + n + m
    return r.x * 100 + r.y * 10 + t + n + m
# INPUTS [{'n': 9, 'm': 0}, {'n': 2, 'm': 1}, {'n': 5, 'm': 4}]
