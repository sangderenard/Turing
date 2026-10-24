def head(xs: list[int]) -> int:
    return xs[0]


def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def root(n: int) -> int:
    # A call that stores into its list argument is a write of that list, so
    # every later read of the list follows the call, even when table and
    # comprehension work pull the reading region elsewhere.
    d0 = {0: n}
    d0[1] = d0.get(0, -1) + 5
    t1 = 0
    for y1 in [x1 * n for x1 in [6, 0]]:
        t1 += y1
    xs2 = [n, 6]
    h2 = head(xs2)
    swap(xs2)
    return (d0[1] * 10 + len(d0)) + t1 + (h2 * 100 + xs2[0] * 10 + xs2[1])

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
