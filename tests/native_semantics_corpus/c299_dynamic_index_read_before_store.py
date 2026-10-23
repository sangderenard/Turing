def root(n: int) -> int:
    xs = [3, n]
    xs.append(1)
    a = xs[n % 3]
    xs[n % 3] = 8
    return a * 10 + xs[n % 3]
# INPUTS [{"n":0},{"n":2},{"n":-3}]
