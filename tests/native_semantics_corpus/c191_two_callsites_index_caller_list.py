def pick(xs: list[int], i: int) -> int:
    return xs[i]


def root(n: int) -> int:
    xs = [n, 4, 9]
    return pick(xs, n % 3) * 10 + pick(xs, 2)

# INPUTS [{'n': 1}, {'n': 5}]
