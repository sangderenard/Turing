def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def root(n: int) -> int:
    xs = [n, 4]
    swap(xs)
    swap(xs)
    swap(xs)
    return xs[0] * 10 + xs[1]

# INPUTS [{'n': 1}, {'n': 5}]
