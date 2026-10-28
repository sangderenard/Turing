def total(d: dict[int, int]) -> int:
    t = 0
    for k, v in d.items():
        t += k * v
    return t


def root(n):
    return total({1: n, 2: 5, 3: 7})
# INPUTS [{"n": 0}, {"n": 4}, {"n": -3}]
