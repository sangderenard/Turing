def root(n: int) -> int:
    xs = [5, n]
    xs.extend([7, 8])
    return xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + xs[3]

# INPUTS [{'n': 1}, {'n': 4}]
