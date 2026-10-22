def root(n: int) -> int:
    c0 = abs(n - 3)
    xs1 = [6, 5, n]
    xs1.reverse()
    return c0 + xs1[0]

# INPUTS [{'n': 3}, {'n': -5}, {'n': 9}]
