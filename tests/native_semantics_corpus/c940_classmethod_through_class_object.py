# A class method called through its class, and through cls inside another
# class method: the class object binds as cls.  The prepass admitted only
# static methods there and refused (bootstrap: CorrelationTable.build).
class Scale:
    def __init__(self, k: int) -> None:
        self.k = k

    @classmethod
    def apply(cls, n: int) -> int:
        return n * 3 + 1

    @classmethod
    def twice(cls, n: int) -> int:
        return cls.apply(cls.apply(n))


def root(n: int) -> int:
    return Scale.apply(n) * 100 + Scale.twice(n)

# INPUTS [{'n': 0}, {'n': 5}, {'n': -2}]
