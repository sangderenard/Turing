class Index:
    def __init__(self):
        self._candidates = {}

    def batch(self, a: int, b: int, c: int, d: int, token: int):
        self._candidates.setdefault((a, b, c, d), []).append(token)
        return 1

def root(a: int, b: int, c: int, d: int, token: int):
    index = Index()
    return index.batch(a, b, c, d, token)
