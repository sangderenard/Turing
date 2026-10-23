def root(n: int) -> int:
    xs = [4] * 3
    ys = [0] * (n % 3 + 1)
    ys[0] = n
    return len(xs) * 100 + xs[2] * 10 + len(ys) + ys[0]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
