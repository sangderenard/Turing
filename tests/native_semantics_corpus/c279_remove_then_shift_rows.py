def root(n: int) -> int:
    xs = [4, n, 7, 1]
    xs.remove(n)
    return len(xs) * 1000 + xs[0] * 100 + xs[1] * 10 + xs[2]

# INPUTS [{'n': 0}, {'n': 4}, {'n': 7}]
