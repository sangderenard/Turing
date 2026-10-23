def root(n: int) -> int:
    xs = [7, 8]
    t = 0
    for i in range(n):
        t += xs[0] + len(xs)
        xs = [xs[1] + n, i, 1]
    return t * 100 + xs[0] * 10 + len(xs)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
