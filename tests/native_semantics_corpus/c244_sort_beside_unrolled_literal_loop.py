def root(n: int) -> int:
    t = 0
    for x in [3, -2, 0]:
        t += x * n
    xs = [6, 7, n]
    xs.sort()
    ys = [y * n for y in [3, 9, 0]]
    zs = [2, n, 9, 0]
    zs.sort(reverse=True)
    return (xs[0] - xs[2]) * 1000 + zs[0] * 100 + zs[3] * 10 + sum(ys) + t

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}, {'n': 12}]
