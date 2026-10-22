def root(n: int) -> float:
    xs = [0.5 * n, 2.0, -1.5]
    xs.append(n * 1.25)
    return max(xs) + min(xs) * 10.0 + sum(xs) * 100.0

# INPUTS [{'n': 1}, {'n': -3}]
