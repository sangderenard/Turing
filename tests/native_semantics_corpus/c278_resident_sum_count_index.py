def root(n: int) -> int:
    xs = [6, 3, 6, n]
    return sum(xs) * 1000 + xs.count(6) * 100 + xs.index(n) * 10 + xs.index(3)

# INPUTS [{'n': 0}, {'n': 6}, {'n': 3}]
