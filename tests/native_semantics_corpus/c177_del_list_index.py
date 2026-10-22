def root(n: int) -> int:
    xs = [5, n, 9, 8, 7]
    del xs[-2]
    del xs[0]
    return len(xs) * 1000 + xs[0] * 100 + xs[1] * 10 + xs[2]

# INPUTS [{'n': 1}, {'n': 4}]
