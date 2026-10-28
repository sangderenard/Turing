# ``.get`` on a dict field reads the entry stored earlier in the same call,
# with a computed key, and falls back to its default when the key is absent.
class Tally:
    def __init__(self):
        self._d = {}

    def put(self, k: int, v: int) -> int:
        self._d[k] = v
        return self._d.get(k + 7, -1) + len(self._d)


def root(t: int) -> int:
    b = Tally()
    b.put(t, 1)
    b.put(t + 7, 2)
    return b.put(t, 5) * 10 + b.put(t + 20, 3)
# INPUTS [{'t': 3}, {'t': -2}]
