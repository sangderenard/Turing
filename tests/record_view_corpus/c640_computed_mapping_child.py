from array import array

AGGREGATE_BASE = 1114112


class Index:
    def __init__(self):
        self.domain = array("B")
        self._candidates = {}
        self._prefix = {}

    def put(self, domain: int, hi: int, lo: int, atom_count: int, atom: int):
        token = AGGREGATE_BASE + len(self.domain)
        self.domain.append(domain)
        key = (domain, hi, lo, atom_count)
        self._candidates.setdefault(key, []).append(token)
        self._prefix.setdefault((domain, atom), []).append(token)

    def read(self, domain: int, hi: int, lo: int, atom_count: int, atom: int):
        total = 0
        key = (domain, hi, lo, atom_count)
        for token in self._candidates.get(key, ()):
            total += token
        for token in self._prefix.get((domain, atom), ()):
            total += token
        return total


def root(index: Index, domain: int, hi: int, lo: int, atom_count: int, atom: int):
    index.put(domain, hi, lo, atom_count, atom)
    return index.read(domain, hi, lo, atom_count, atom)
