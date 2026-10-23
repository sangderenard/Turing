def root(n: int) -> int:
    xs = [3, n, 4, 7]
    ys = xs[-3:-1]
    zs = xs[1:]
    return ys[0] * 10 + ys[1] + len(ys) * 100 + zs[2] * 1000
# INPUTS [{'n':4},{'n':-1},{'n':2}]
