def root(n: int) -> int:
    xs = [5, n, 6]
    t = 0
    for k, x in enumerate(xs, start=2):
        t += k * x
    return t

# INPUTS [{'n': 2}, {'n': -1}]
