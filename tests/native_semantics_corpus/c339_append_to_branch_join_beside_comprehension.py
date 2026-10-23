def root(n: int) -> int:
    if n > 6:
        t0 = [n, 3]
    else:
        t0 = [8, n]
    t0.append(6)
    s1 = {(x1 + n) % 3 for x1 in [5, 0, 7]}
    ys = [x * n for x in [2, 5]]
    return len(t0) * 100 + t0[0] * 10 + t0[2] + len(s1) * 1000 + sum(ys) * 10000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 7}]
