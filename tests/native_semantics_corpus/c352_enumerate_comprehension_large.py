def root(n: int) -> int:
    xs = [5, n + 36028797018963969, 6]
    ys = [k * x for k, x in enumerate(xs)]
    return sum(ys)

# INPUTS [{'n': 2}, {'n': -7}]
