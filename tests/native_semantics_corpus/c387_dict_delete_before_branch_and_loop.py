def maybe_pick(x: int, y: int):
    return x * 2 if x != y else None


def root(n: int) -> int:
    # A ``del`` outside every branch and loop runs once, at its place, even
    # when the next region is a conditional arm or a loop body.
    o1 = maybe_pick(n + 1, 1)
    d2 = {1: 5, 2: n, 3: 8}
    del d2[2]
    a = (-3 if o1 is None else o1 + 1) + len(d2) * 10
    d1 = {1: n, 2: 5}
    del d1[1]
    t2 = 0
    for y2 in [x2 * n for x2 in [3, -1]]:
        t2 += y2
    b = len(d1) * 10 + int(1 in d1) + d1[2] + t2
    return a * 1000 + b

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
