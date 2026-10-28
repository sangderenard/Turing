def same(a: list[int]) -> list[int]:
    return a


class Bag:
    def __init__(self, c: list[int]):
        self._c = c

    def add(self, x: int):
        self._c.append(x)


def root(n: int) -> int:
    a = [n, 2, 3]
    b = Bag(same(a))
    b.add(n)
    return len(a) * 10 + a[3]
# INPUTS [{'n': 0}, {'n': 4}]
