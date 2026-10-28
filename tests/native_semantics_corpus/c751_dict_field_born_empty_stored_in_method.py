# A dict field born empty in __init__ and written by key in a method keeps
# its rows across calls; repeated keys overwrite.
class Tally:
    def __init__(self):
        self._d = {}

    def put(self, k: int, v: int) -> int:
        self._d[k] = v
        return len(self._d) * 100 + self._d[k]


def root(t: int) -> int:
    b = Tally()
    b.put(t, 1)
    b.put(t + 1, 2)
    return b.put(t, 5)
# INPUTS [{'t': 3}, {'t': -2}]
