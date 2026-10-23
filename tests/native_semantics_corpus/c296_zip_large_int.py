def root(n: int) -> int:
    xs = [1, n, 3]
    ys = [n, 5, 1]
    t = 0
    for p, q in zip(xs, ys):
        t += p * 3 + q
    return t
# INPUTS [{'n':4},{'n':36028797018963969}]
