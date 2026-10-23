def root(n: int) -> int:
    # The loop reads its enumerate start, so the start is computed before
    # the loop even when other statements pull its region later.
    xs0 = [4, n, 6]
    r0 = xs0[-1]
    xs0.pop(0)
    t2 = 0
    for k2, x2 in enumerate([5, n, 1], n % 4 + 4):
        t2 = t2 * 10 + k2 * x2
    u = 0
    for i in range(2):
        for k3, x3 in enumerate([n, 2], i + n % 3):
            u += k3 * x3
    return r0 * 10 + len(xs0) + t2 + u * 100000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
