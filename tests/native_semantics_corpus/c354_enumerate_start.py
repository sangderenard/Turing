def root(n: int) -> int:
    xs = [5, n + 36028797018963969, 6]
    t = 0
    for k, x in enumerate(xs, 3):
        t += k * x
    return t

# INPUTS [{'n': 2}, {'n': -7}]
