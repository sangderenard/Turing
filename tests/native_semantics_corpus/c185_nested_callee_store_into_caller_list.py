def inner(xs: list[int]) -> None:
    xs[1] = 7


def outer(xs: list[int]) -> int:
    inner(xs)
    return xs[0]


def root(n: int) -> int:
    xs = [n, 4]
    a = outer(xs)
    return a * 100 + xs[1] * 10 + len(xs)

# INPUTS [{'n': 1}, {'n': 5}]
