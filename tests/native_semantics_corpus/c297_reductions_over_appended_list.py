def root(n: int) -> int:
    xs = [n, 3, n * n, -7]
    xs.append(n - 4)
    return max(xs) * 1000 + min(xs) * 10 + sum(xs, n)

# INPUTS [{'n': 0}, {'n': 5}, {'n': -6}, {'n': 2}]
