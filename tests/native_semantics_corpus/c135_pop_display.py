def root(n: int) -> int:
    xs = [5, n, 2]
    xs.pop()
    return xs[0] * 10 + xs[1] + len(xs) * 100

# INPUTS [{'n': 1}, {'n': 4}]
