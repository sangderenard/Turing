def root(n: int) -> int:
    xs0 = [1, n, 4]
    ys0 = [n, 5, 1]
    t0 = 0
    for p0, q0 in zip(xs0, ys0):
        t0 += p0 * q0
    return (t0)
# INPUTS [{'n': 2}, {'n': -3}]
