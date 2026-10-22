def root(n: int) -> int:
    xs = [1, n, 4, 7]
    ys = [n, 5]
    t = 0
    for p, q in zip(xs, ys):
        t += p * q + 1
    return t

# INPUTS [{'n': 2}, {'n': -3}]
