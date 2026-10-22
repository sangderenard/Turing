def root(n: int) -> int:
    xs = [1, 4, 9]
    ys = [x * n for x in xs]
    return sum(ys) * 10 + sum([x for x in xs])

# INPUTS [{'n': 0}, {'n': 1}, {'n': -3}, {'n': 7}]
