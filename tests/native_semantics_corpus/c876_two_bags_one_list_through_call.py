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
    c = Bag(a)
    c.add(n)
    b.add(2)
    a.append(1)
    return len(a) * 1000 + b.total() * 10 + c.total() - a[0]
# INPUTS [{'n': 0}, {'n': 4}]
