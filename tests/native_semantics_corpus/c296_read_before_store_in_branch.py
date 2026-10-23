def root(n: int) -> int:
    xs = [3, n]
    xs.append(1)
    a = xs[1]
    if n > 0:
        xs[1] = 7
    return a * 10 + xs[1]
# INPUTS [{"n":0},{"n":2},{"n":-3}]
