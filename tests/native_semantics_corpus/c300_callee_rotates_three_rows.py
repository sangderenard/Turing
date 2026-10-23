def rotate(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = xs[2]
    xs[2] = t


def root(n: int) -> int:
    xs = [n, 4, 9]
    rotate(xs)
    rotate(xs)
    return xs[0] * 100 + xs[1] * 10 + xs[2]

# INPUTS [{'n': 1}, {'n': 7}, {'n': -2}]
