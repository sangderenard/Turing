# The dict-of-lists field read-back with a tuple key.
class S:
    def __init__(self):
        self.d = {}

    def put(self, a: int, b: int, v: int):
        self.d.setdefault((a, b), []).append(v)

    def first(self, a: int, b: int) -> int:
        for t in self.d.get((a, b), ()):
            return t
        return 0


def root(n: int) -> int:
    s = S()
    s.put(n, 1, 7)
    return s.first(n, 1) * 10 + s.first(n, 2)
# INPUTS [{'n': 3}, {'n': -1}]
