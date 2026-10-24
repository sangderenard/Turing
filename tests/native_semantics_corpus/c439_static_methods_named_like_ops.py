class Ops:
    # Static methods spelled like canonical tensor ops, called through the
    # class: each is the authored body, never the op.
    @staticmethod
    def min(a: int, b: int) -> int:
        return a - b

    @staticmethod
    def sum(a: int) -> int:
        return a * 3 + 1

    @staticmethod
    def scale(a: int, k: int) -> int:
        return a * k


def root(n: int) -> int:
    return Ops.min(n, 1) * 100 + Ops.sum(n) * 10 + Ops.scale(n, 2)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 7}]
