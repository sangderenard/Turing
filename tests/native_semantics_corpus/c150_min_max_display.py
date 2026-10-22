def root(n: int) -> int:
    xs = [n, 7, -2, n * 2]
    return max(xs) * 100 + min(xs)

# INPUTS [{'n': 0}, {'n': 5}, {'n': -4}]
