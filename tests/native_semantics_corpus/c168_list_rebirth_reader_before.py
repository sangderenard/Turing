def root(n: int) -> int:
    xs = [n, 2]
    a = xs[0] + xs[1]
    xs = [a, n * 2]
    return a * 100 + xs[0] * 10 + xs[1]

# INPUTS [{'n': 1}, {'n': 4}]
