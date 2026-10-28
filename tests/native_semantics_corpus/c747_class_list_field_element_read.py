# A list field born empty and grown by a method is read back by index in the
# same method; the read sees the row stored by the first call.
class Bag:
    def __init__(self):
        self._c = []

    def put(self, t: int) -> int:
        self._c.append(t)
        return self._c[0] * 10 + self._c[-1]


def root(t: int) -> int:
    b = Bag()
    b.put(t)
    return b.put(t + 1)
# INPUTS [{'t': 3}, {'t': -5}]
