class S:
    def __init__(self):
        self.d = {}

    def put(self, k: int, v: int):
        self.d.setdefault(k, []).append(v)

    def get(self, k: int) -> int:
        for t in self.d.get(k, ()):
            return t
        return 0


def root(n: int):
    s = S()
    s.put(n, 7)
    return s.get(n) * 10 + s.get(n + 1)

# INPUTS [{'n': 3}]
