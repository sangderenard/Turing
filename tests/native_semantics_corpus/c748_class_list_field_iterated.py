# A list field born empty is iterated in the method that grows it; the loop
# target takes the field's element class.
class Bag:
    def __init__(self):
        self._c = []

    def put(self, t: float) -> float:
        self._c.append(t)
        s = 0.0
        for v in self._c:
            s += v
        return s + len(self._c)


def root(t: float) -> float:
    b = Bag()
    b.put(t)
    b.put(t * 0.5)
    return b.put(t + 1.0)
# INPUTS [{'t': 3.0}, {'t': -5.5}]
