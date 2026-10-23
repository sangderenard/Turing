def root(n: int) -> int:
    xs = [3, n]
    xs.append(1)
    a = xs[1]
    if n > 0:
        xs.pop(1)
    return a * 10 + len(xs)
# INPUTS [{"n":0},{"n":2},{"n":-3}]
