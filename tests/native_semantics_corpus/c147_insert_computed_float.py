def root(n: int) -> float:
    xs = [0.5, 1.5]
    xs.insert(1, n * 0.25)
    xs.reverse()
    return xs[0] + xs[1] * 10.0 + xs[2] * 100.0

# INPUTS [{'n': 2}, {'n': 3}]
