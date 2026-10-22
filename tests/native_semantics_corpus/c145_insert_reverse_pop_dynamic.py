def root(n: int) -> int:
    xs = []
    for i in range(n):
        xs.insert(0, i * 3)
    xs.reverse()
    first = xs.pop(0)
    return first * 1000 + xs[0] * 100 + xs[len(xs) - 1] * 10 + len(xs)

# INPUTS [{'n': 3}, {'n': 6}]
