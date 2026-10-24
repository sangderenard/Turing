def make(k: int):
    def f(x: int) -> int:
        return x + k
    return f


def root(n: int) -> int:
    g = make(n)
    return g(3)
# INPUTS [{"n":0},{"n":4}]
