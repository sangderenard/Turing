def root(n: int) -> int:
    xs = [n, 3, 2]
    xs[0], xs[2] = xs[2], xs[0]
    return xs[0] * 100 + xs[1] * 10 + xs[2]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}]
