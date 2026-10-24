def root(n: int) -> int:
    xs = [0, 0, 0, 0]
    for i in range(6):
        xs[i % 4] += i + n
    ys = [1, 2, 3]
    ys[n % 3] += 10
    ys[(n + 1) % 3] *= 2
    zs = [5, 6]
    for i in range(5):
        k = (i + n) % 2
        zs[k] -= i
    return (
        xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + xs[3]
        + ys[0] * 7 + ys[1] * 5 + ys[2] * 3
        + zs[0] * 11 + zs[1]
    )

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
