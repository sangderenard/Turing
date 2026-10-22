def root(n: int) -> int:
    k = n + 1
    xs = [4, k, 2]
    ys = [k, 9]
    return xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + ys[0] + len(ys) * 10000

# INPUTS [{'n': 1}, {'n': 7}]
