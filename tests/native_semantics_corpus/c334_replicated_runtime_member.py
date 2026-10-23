def root(n: int) -> int:
    xs = [n] * 3
    xs[1] = 3
    return sum(xs) * 10 + xs[1] + len(xs) * 1000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
