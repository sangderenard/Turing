def addto(s: set[int], v: int) -> None:
    s.add(v)


def root(n: int) -> int:
    # A set display and a list display hold the same value ``n``; each
    # display's member births are their own, so both lists are born and
    # the program is lowered rather than refused.
    s0 = {n, 5}
    addto(s0, 5)
    addto(s0, n)
    s0.add(7)
    xs1 = [6, n, -3]
    ys1 = xs1[-2:]
    xs1.append(1)
    return len(s0) * 100 + sum(s0) + ys1[0] * 10 + len(ys1) + xs1[1] + len(xs1) * 1000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}, {'n': 7}]
