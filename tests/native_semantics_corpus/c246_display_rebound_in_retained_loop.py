def root(n: int) -> int:
    xs = [0]
    t = 0
    for i in range(4):
        t += len(xs) * 100 + xs[-1]
        xs = [i, n, i + 1]
    return t * 10 + len(xs)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}]
