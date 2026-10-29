def pick(d: dict[int, int]) -> int:
    return d[2] + len(d)


def root(n):
    return pick({1: n, 2: 5, 3: 7})
# INPUTS [{"n": 0}, {"n": 4}, {"n": -3}]
