def root(n: int) -> int:
    xs = [5, n, 6]
    t = 0
    for k, x in enumerate(xs, n):
        t = t * 10 + k * x
    return t

# INPUTS [{'n': 2}, {'n': -1}, {'n': 4}]
