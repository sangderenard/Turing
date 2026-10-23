def grow(x: int) -> int:
    xs = [7, 5]
    xs.append(x)
    return xs[0] + xs[2] * 10


def root(n: int) -> int:
    ys = [5, n, 4]
    ys[0] += n
    c = grow(n + 6)
    return ys[0] + c

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}]
