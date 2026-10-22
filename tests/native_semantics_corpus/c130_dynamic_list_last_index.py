def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append((i * 7) % 5)
    return xs[0] * 100 + xs[len(xs) - 1]

# INPUTS [{'n': 3}, {'n': 6}]
