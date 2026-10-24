class Ops:
    @staticmethod
    def min(a: int, b: int) -> int:
        return a - b


def clamp(n: int) -> int:
    return min(n, 3)


def root(n: int) -> int:
    probe = Ops.min
    return clamp(n) * 100 + n
# INPUTS [{"n":0},{"n":2},{"n":7}]
