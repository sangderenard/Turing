def root(n: int) -> int:
    xs = [3, 1, 4, 7]
    ys = xs[2:]
    return sum(xs) * 100 + sum(ys) * 10 + n
# INPUTS [{'n':4},{'n':-1}]
