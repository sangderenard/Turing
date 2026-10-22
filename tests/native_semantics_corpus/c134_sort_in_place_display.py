def root(n: int) -> int:
    xs = [5, n, 2, 9]
    xs.sort()
    return xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + xs[3]

# INPUTS [{'n': 0}, {'n': 7}]
