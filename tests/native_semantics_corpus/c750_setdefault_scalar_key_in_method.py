# setdefault under a plain int key on a dict field inside a method: the key
# is one column whose class is the key's own.
class Index:
    def __init__(self):
        self._d = {}

    def batch(self, k: int, token: int) -> int:
        self._d.setdefault(k, []).append(token)
        return len(self._d)


def root(k: int, token: int) -> int:
    index = Index()
    index.batch(k, token)
    index.batch(k + 1, token)
    index.batch(k, token + 1)
    return index.batch(k + 2, token) * 10
# INPUTS [{'k': 1, 'token': 5}, {'k': -4, 'token': 0}]
