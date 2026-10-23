def head(xs: list[int]) -> int:
    return xs[0]


def width(xs: list[int]) -> int:
    return len(xs)


def root(n: int) -> int:
    # head() sees one list bound by its display members and one grown list
    # bound as storage; the two calls need two different frames.
    xs0 = [n, 1]
    xs2 = [n, 2, 6]
    xs2.append(0)
    first = head(xs0) * 100 + head(xs2) * 10 + width(xs2)
    ys2 = [n + 1, 5]
    ys2.append(n)
    ys0 = [n - 2, 4]
    return first * 1000 + head(ys2) * 100 + head(ys0) * 10 + width(ys0)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
