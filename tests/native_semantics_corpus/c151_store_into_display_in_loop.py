def root(n: int) -> int:
    xs = [0, 0, 0, 0]
    for i in range(4):
        xs[i] = i * n
    xs[2] += 5
    return xs[0] + xs[1] * 10 + xs[2] * 100 + xs[3] * 1000

# INPUTS [{'n': 1}, {'n': 3}]
