def root(n: int) -> int:
    xs = [3, n, 4, 7]
    ys = xs[1:3]
    ys.append(n)
    zs = xs[:2]
    xs.append(n)
    return ys[2] * 10 + len(ys) + len(zs) * 100 + len(xs) * 1000
# INPUTS [{'n':4},{'n':-1},{'n':2}]
