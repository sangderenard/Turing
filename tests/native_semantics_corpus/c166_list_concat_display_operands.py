def root(n: int) -> int:
    xs0 = [5, 7] + [n, 7]
    return len(xs0) * 10 + xs0[2]

# INPUTS [{'n': 3}, {'n': -3}]
