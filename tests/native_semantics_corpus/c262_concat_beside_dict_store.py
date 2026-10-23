def root(n: int) -> int:
    d = {1: 5, 2: 4}
    d[3] = n
    xs = [1, 2] + [n, 8]
    return xs[2] * 100 + xs[3] * 10 + len(xs) + d[3] * 1000
# INPUTS [{'n':3},{'n':7},{'n':-1}]
