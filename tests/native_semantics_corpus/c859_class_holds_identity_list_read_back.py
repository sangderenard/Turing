def same(a: list[int]) -> list[int]:
    return a


class Bag:
    def __init__(self, c: list[int]):
        self._c = c

    def add(self, x: int):
        self._c.append(x)

    def total(self) -> int:
        return sum(self._c)


def root(n: int) -> int:
    a = [n, 2, 3]
    b = Bag(same(a))
    b.add(n)
    a.append(1)
    return b.total() * 10 + len(a)
# INPUTS [{'n': 0}, {'n': 4}]
