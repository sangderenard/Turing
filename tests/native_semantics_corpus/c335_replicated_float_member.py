def root(n: int) -> float:
    x = n * 0.5
    xs = [x] * 3
    xs[0] = 1.25
    return sum(xs) + xs[2]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
