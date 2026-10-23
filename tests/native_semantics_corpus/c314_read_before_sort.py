def root(n: int) -> int:
    xs = [5, n, 1]
    xs.append(2)
    a = xs[0]
    xs.sort()
    return a * 100 + xs[0] * 10 + xs[3]
# INPUTS [{"n":0},{"n":2},{"n":-3}]
