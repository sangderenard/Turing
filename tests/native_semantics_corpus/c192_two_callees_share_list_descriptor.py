def pick(xs: list[int], i: int) -> int:
    return xs[i]


def width(xs: list[int]) -> int:
    return len(xs)


def root(n: int) -> int:
    xs = [n, 5, 9]
    return pick(xs, n % 3) * 10 + width(xs)

# INPUTS [{'n': 1}, {'n': 5}]
