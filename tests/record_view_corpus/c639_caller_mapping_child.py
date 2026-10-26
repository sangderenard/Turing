class Index:
    def __init__(self):
        self.table = {}

    def put(self, key: int, value: int):
        self.table.setdefault(key, []).append(value)

    def read(self, key: int):
        total = 0
        for value in self.table.get(key, ()):
            total += value
        return total


def root(index: Index, write: bool, key: int, value: int, query: int):
    if write:
        index.put(key, value)
    return index.read(query)
