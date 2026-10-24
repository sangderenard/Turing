def make(k: int):
    def f(x: int) -> int:
        return x + k
    return f


def root(n: int) -> int:
    return make(n)(3) + make(1)(n)
# INPUTS [{"n":0},{"n":4}]
