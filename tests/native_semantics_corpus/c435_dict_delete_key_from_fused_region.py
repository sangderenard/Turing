def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def head(xs: list[int]) -> int:
    return xs[0]


def root(n: int) -> int:
    # The delete's computed key shares a region with an earlier list read;
    # the delete runs after that region defines the key.
    xs0 = [n, 3]
    r0 = xs0[0]
    swap(xs0)
    d1 = {1: 2, 2: n}
    del d1[n % 3 + 1]
    return (r0 * 100 + head(xs0) * 10 + xs0[1]) + (len(d1) * 10 + d1.get(1, 3))

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 4}]
