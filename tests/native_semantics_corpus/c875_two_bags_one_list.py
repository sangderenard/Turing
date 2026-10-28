class Bag:
    def __init__(self, c: list[int]):
        self._c = c

    def add(self, x: int):
        self._c.append(x)

    def total(self) -> int:
        return sum(self._c)


def root(n: int) -> int:
    a = [n, 2, 3]
    b = Bag(a)
    c = Bag(a)
    b.add(n)
    c.add(1)
    return len(a) * 100 + b.total()
# INPUTS [{'n': 0}, {'n': 4}]
