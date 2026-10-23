def root(n: int) -> int:
    xs = [1, 4, 9]
    ys = [x * n for x in xs]
    flags = [x for x in [True, False, True]]
    return ys[n % 3] * 1000 + len(ys) * 100 + sum(flags) * 10 + n

# INPUTS [{'n': 0}, {'n': 4}, {'n': 5}, {'n': -2}]
