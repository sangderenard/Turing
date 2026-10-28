# An annotated dict field the initializer births empty is the class's own
# growable mapping: a method stores into it, reads it back, counts it and
# iterates it.
class Tally:
    def __init__(self):
        self._d: dict[int, float] = {}

    def put(self, k: int, v: float) -> float:
        self._d[k] = v
        s = 0.0
        for key in self._d:
            s += self._d[key] * key
        return s + self._d[k] + len(self._d)


def root(t: int) -> float:
    b = Tally()
    b.put(t, 1.5)
    b.put(t - 1, 2.25)
    return b.put(t, 0.5)
# INPUTS [{'t': 3}, {'t': -2}]
