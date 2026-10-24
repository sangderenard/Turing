def make(n: int):
    total = 0

    def f(x: int) -> int:
        return x + total

    for i in range(n):
        total = total + i
    return f


def root(n: int) -> int:
    g = make(n)
    return g(100)
# INPUTS [{"n":0},{"n":4}]
