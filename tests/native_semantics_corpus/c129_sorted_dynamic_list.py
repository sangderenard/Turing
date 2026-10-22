def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.append((i * 7) % 5)
    ys = sorted(xs)
    return ys[0] * 100 + ys[len(ys) - 1]

# INPUTS [{'n': 3}, {'n': 6}]
