# A dict-of-lists field grown by ``setdefault(k, []).append(v)`` in one
# method reads back through ``.get(k, ())`` and ``.get(k, [])`` in another;
# a missing key yields the empty default, never child row zero.
class S:
    def __init__(self):
        self.d = {}

    def put(self, k: int, v: int):
        self.d.setdefault(k, []).append(v)

    def first(self, k: int) -> int:
        for t in self.d.get(k, ()):
            return t
        return 0

    def total(self, k: int) -> int:
        s = 0
        for t in self.d.get(k, []):
            s += t
        return s


def root(n: int) -> int:
    s = S()
    s.put(n, 7)
    s.put(n, 4)
    return s.first(n) * 1000 + s.first(n + 1) * 100 + s.total(n) * 10 + s.total(n + 1)
# INPUTS [{'n': 3}, {'n': -1}]
