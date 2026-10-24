def addto(s: set[int], v: int) -> None:
    s.add(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    # A region that reduces an arena runs after the call that stores into
    # it, even when other loop work fills the schedule ahead of both.
    xs0 = [2, 0, 4]
    for i0 in range(3):
        xs0[i0] += 1
    s2 = {n, 6}
    addto(s2, 4)
    ys = [n, 6]
    setfirst(ys, 4)
    return xs0[0] + sum(s2) * 10 + sum(ys) * 1000 + max(ys) * 100000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}, {'n': 9}]
