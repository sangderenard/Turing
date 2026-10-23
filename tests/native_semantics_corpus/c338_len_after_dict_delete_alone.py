def root(n: int) -> int:
    d = {1: n, 2: 3, 5: 7}
    del d[2]
    return len(d) * 100 + d[5] * 10 + n

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
