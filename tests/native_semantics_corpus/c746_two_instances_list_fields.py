# Two instances of one class own separate list fields.
class Bag:
    def __init__(self):
        self._c = []

    def put(self, t: int) -> int:
        self._c.append(t)
        return len(self._c)


def root(t: int) -> int:
    a = Bag()
    b = Bag()
    a.put(t)
    a.put(t)
    return a.put(t) * 10 + b.put(t)
# INPUTS [{'t': 1}, {'t': 9}]
