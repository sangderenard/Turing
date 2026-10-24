def f(n: int) -> int:
    xs = [7, 5]
    xs.append(n)
    return xs[0] * 10 + xs[2]


def root(n: int) -> int:
    return f(n) * 1000 + f(n + 1)

# INPUTS [{"n": 0}, {"n": 3}]
