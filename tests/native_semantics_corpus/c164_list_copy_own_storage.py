def root(n: int) -> int:
    xs0 = [1, n, 5]
    ys0 = list(xs0)
    ys0.append(n)
    return len(ys0) * 10 + len(xs0)

# INPUTS [{'n': 3}, {'n': -1}]
