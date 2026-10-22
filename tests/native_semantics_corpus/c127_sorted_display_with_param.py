def root(n: int) -> int:
    xs = [5, n, 2, 9]
    ys = sorted(xs)
    return ys[0] * 1000 + ys[1] * 100 + ys[2] * 10 + ys[3]

# INPUTS [{'n': 0}, {'n': 7}]
