def bump(d: dict[int, int], n: int) -> int:
    d[0] += n
    return d[0]


def bump_back(d: dict[int, int], n: int) -> int:
    d[1] -= n
    return d[1] * 2


def root(n: int) -> int:
    a = bump({0: 1, 1: 3}, n)
    b = bump_back({0: 4, 1: 7}, n)
    return a * 100 + b
# INPUTS [{"n": -3}, {"n": 0}, {"n": 2}, {"n": 5}]
