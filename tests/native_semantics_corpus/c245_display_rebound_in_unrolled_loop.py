def root(n: int) -> int:
    xs = [0, 0]
    for i in range(4):
        xs = [i + n, i]
    ys = [0]
    for j in range(3):
        ys = [j + n, j, n]
    return len(xs) * 1000 + xs[0] * 100 + len(ys) * 10 + ys[2] + ys[0]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}]
