def root(n: int) -> int:
    xs = [5, n, 2]
    v = xs.pop()
    return v * 100 + xs[0] * 10 + xs[1]

# INPUTS [{'n': 1}, {'n': 4}]
