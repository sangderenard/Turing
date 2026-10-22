def root(n: int) -> int:
    xs = [5] * (n % 4 + 1)
    return len(xs) * 10 + xs[0]

# INPUTS [{'n': 1}, {'n': 5}, {'n': 2}]
