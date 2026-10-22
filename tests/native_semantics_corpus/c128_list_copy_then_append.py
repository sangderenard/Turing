def root(n: int) -> int:
    xs = [5, n, 2, 9]
    ys = list(xs)
    ys.append(n)
    return ys[0] * 1000 + ys[1] * 100 + ys[4]

# INPUTS [{'n': 0}, {'n': 7}]
