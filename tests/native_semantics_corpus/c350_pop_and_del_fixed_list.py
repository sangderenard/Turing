def root(n: int) -> int:
    xs = [3, n, 8, 1, n + 5]
    a = xs.pop(2)
    del xs[0]
    b = xs.pop(-1)
    return a * 1000 + b * 100 + xs[0] * 10 + xs[1]

# INPUTS [{'n': 2}, {'n': 4}, {'n': 0}]
