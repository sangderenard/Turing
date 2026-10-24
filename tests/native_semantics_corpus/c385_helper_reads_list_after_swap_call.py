def head(xs: list[int]) -> int:
    return xs[0]


def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def root(n: int) -> int:
    # ``head`` reads the list after ``swap`` stored into it, so it is bound
    # to the list's storage, not to the display's members; the read of
    # ``xs[0]`` before the call keeps the value from before the swap.
    xs = [n, 6]
    a = xs[0]
    swap(xs)
    t = 0
    for y in [x * n for x in [6, 0]]:
        t += y
    ys = [n, 2]
    swap(ys)
    return t + a * 100 + head(xs) * 10 + head(ys) * 1000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
