def root(n: int) -> int:
    xs = [6, 3, 6, n]
    return xs.count(6) * 10 + xs.index(3) + xs.index(n) * 100

# INPUTS [{'n': 0}, {'n': 6}, {'n': 3}]
