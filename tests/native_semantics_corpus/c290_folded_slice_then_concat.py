def root(n: int) -> int:
    xs = [3, 1, 4, 1, 5]
    ys = xs[-4:-1] + [n, 9]
    return len(ys) * 100 + ys[0] * 10 + ys[3] * 7 + ys[-1]
# INPUTS [{'n':4},{'n':-1},{'n':0}]
