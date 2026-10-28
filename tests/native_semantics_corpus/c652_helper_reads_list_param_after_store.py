def last(xs: list[int]) -> int:
    return xs[1]


def bump(xs: list[int]) -> None:
    xs[1] = xs[1] + 5


def root(n: int) -> int:
    ys = [n, 3]
    bump(ys)
    return last(ys) * 10 + ys[0]

# INPUTS [{'n': 0}, {'n': 4}, {'n': -1}]
