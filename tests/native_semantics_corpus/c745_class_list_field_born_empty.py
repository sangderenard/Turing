# A list field born empty in __init__ and grown by a method keeps its rows
# across calls on one instance.
class Bag:
    def __init__(self):
        self._c = []

    def put(self, t: int) -> int:
        self._c.append(t)
        return len(self._c)


def root(t: int) -> int:
    b = Bag()
    b.put(t)
    b.put(t + 1)
    return b.put(t + 2)
# INPUTS [{'t': 5}, {'t': 0}]
