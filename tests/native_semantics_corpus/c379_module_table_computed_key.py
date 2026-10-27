# A module table read at a key the function computes (silent on main: the
# read and the membership test ran at entry, before the key's region, and
# saw key 0 for every input).
OPS = {0: 5, 1: 9}
VALS = [5, 9, 4]


def root(a: int, i: int) -> int:
    hit = 1 if i % 3 in OPS else 0
    return OPS[i % 2] * 1000 + OPS.get(i % 3, 7) * 100 + VALS[i % 3] * 10 + hit + a
# INPUTS [{"a": 3, "i": 0}, {"a": 7, "i": 1}, {"a": -2, "i": 2}, {"a": 4, "i": 5}]
