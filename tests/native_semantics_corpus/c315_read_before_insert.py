def root(n: int) -> int:
    xs = [5, n]
    xs.append(2)
    a = xs[1]
    xs.insert(0, 9)
    return a * 100 + xs[1] * 10 + len(xs)
# INPUTS [{"n":0},{"n":2},{"n":-3}]
