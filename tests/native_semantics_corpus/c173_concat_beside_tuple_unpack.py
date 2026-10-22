def root(n: int) -> int:
    xs1 = [4, 1] + [n, 2]
    t2 = (n, 4, -1)
    q2, r2, w2 = t2
    return (len(xs1) * 10 + xs1[2]) + (q2 + r2 * w2)

# INPUTS [{'n': 3}, {'n': -2}]
