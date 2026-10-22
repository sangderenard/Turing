def root(n: int) -> int:
    xs = [5, n, 2]
    xs.insert(0, 8)
    return xs[0] * 100 + xs[1] * 10 + xs[2]

# INPUTS [{'n': 1}, {'n': 4}]
