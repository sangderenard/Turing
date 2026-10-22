def root(n: int) -> int:
    xs = [1, 2, 3, 4]
    v = xs.pop(-2)
    xs.insert(-1, n)
    xs.insert(100, 9)
    xs.insert(-100, 7)
    return v * 100000 + xs[0] * 10000 + xs[1] * 1000 + xs[2] * 100 + xs[3] * 10 + xs[4] + len(xs) * 1000000

# INPUTS [{'n': 5}, {'n': 8}]
