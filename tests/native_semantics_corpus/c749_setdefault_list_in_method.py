# setdefault on a dict field inside a method grows the child list stored
# under a tuple key; the field's re-read in the method is one table.
class Index:
    def __init__(self):
        self._candidates = {}

    def batch(self, a: int, b: int, token: int) -> int:
        self._candidates.setdefault((a, b), []).append(token)
        return len(self._candidates)


def root(a: int, b: int, token: int) -> int:
    index = Index()
    index.batch(a, b, token)
    index.batch(a, b + 1, token)
    return index.batch(a, b, token + 1)
# INPUTS [{'a': 1, 'b': 2, 'token': 5}, {'a': -3, 'b': 0, 'token': 7}]
