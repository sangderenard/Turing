class Index:
    def __init__(self):
        self._candidates = {}

    def lookup(self, a: int, b: int, c: int, d: int):
        return len(self._candidates.get((a, b, c, d), ()))

    def relay(self, a: int, b: int, c: int, d: int):
        return self.lookup(a, b, c, d)

    def batch(self, a: int, b: int, c: int, d: int, token: int):
        found = self.relay(a, b, c, d)
        self._candidates.setdefault((a, b, c, d), []).append(token)
        return found


def root(a: int, b: int, c: int, d: int, token: int):
    index = Index()
    return index.batch(a, b, c, d, token)
