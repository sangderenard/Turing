# The rows of a dict field born empty take the class of the values its
# methods store.
class Tally:
    def __init__(self):
        self._d = {}

    def put(self, k: int, v: float) -> float:
        self._d[k] = v
        return self._d[k] + len(self._d)


def root(t: int) -> float:
    b = Tally()
    b.put(t, 1.5)
    b.put(t + 1, 2.0)
    return b.put(t, 0.25)
# INPUTS [{'t': 3}, {'t': -2}]
