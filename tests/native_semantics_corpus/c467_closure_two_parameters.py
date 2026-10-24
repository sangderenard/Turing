def make(k: int, m: int):
    def f(x: int, y: int) -> int:
        return x * k - y * m
    return f


def root(n: int) -> int:
    g = make(n, 2)
    return g(3, n)
# INPUTS [{"n":0},{"n":5}]
