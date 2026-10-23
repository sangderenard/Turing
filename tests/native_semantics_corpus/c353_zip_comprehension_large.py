def root(n: int) -> int:
    xs = [5, n + 36028797018963969, 6]
    ys = [2, 1, n]
    zs = [p * q for p, q in zip(xs, ys)]
    return sum(zs)

# INPUTS [{'n': 2}, {'n': -7}]
