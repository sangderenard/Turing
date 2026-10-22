def root(n: int) -> int:
    xs = [n, 3, n]
    return xs[0] * 100 + xs[1] * 10 + xs[2] + len(xs) * 1000

# INPUTS [{'n': 1}, {'n': 7}]
