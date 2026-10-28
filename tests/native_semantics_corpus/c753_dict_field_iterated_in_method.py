# Iterating a dict field born empty yields its keys; each key reads back its
# stored value.
class Tally:
    def __init__(self):
        self._d = {}

    def put(self, k: int, v: int) -> int:
        self._d[k] = v
        s = 0
        for key in self._d:
            s += key * self._d[key]
        return s


def root(t: int) -> int:
    b = Tally()
    b.put(t, 1)
    b.put(t + 1, 2)
    return b.put(t, 5)
# INPUTS [{'t': 3}, {'t': -2}]
