class Index:
    def __init__(self):
        self.table = {}

    def put(self, key: int, value: int):
        self.table.setdefault(key, []).append(value)


def root(key: int, value: int):
    index = Index()
    index.put(key, value)
    return 1
