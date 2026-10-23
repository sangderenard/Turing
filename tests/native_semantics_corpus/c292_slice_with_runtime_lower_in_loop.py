def root(n: int) -> int:
    xs = [3, n, 4, 7]
    t = 0
    for i in range(3):
        ys = xs[i:]
        t = t * 10 + len(ys) + ys[0]
    return t
# INPUTS [{'n':4},{'n':-1},{'n':2}]
