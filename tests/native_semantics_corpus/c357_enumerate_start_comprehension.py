def root(n: int) -> int:
    xs = [5, n, 6]
    return sum([k * x for k, x in enumerate(xs, 2)])

# INPUTS [{'n': 2}, {'n': -1}]
