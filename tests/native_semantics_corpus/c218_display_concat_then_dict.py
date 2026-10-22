def root(n: int) -> int:
    xs = [1, -1] + [n, 1]
    d = {1: 6}
    ys = [n * 2] + [3] + [n, n + 1]
    return xs[2] * 10 + d[1] + ys[3] * 1000 + len(ys) * 100000

# INPUTS [{'n': -3}, {'n': 2}, {'n': 0}, {'n': 7}]
