# Token Orchard's ``_intern_validated_span``, reduced: ``intern`` takes an
# optional preflight ``fingerprint`` that its one caller, ``intern_one``,
# never passes (default ``None``), so ``hi, lo = fingerprint`` never runs and ``hi, lo``
# are exactly the uint64 pair ``fingerprint`` computes.  The dict-of-lists
# key columns ``(domain, hi, lo, n)`` take that width.  Keys never repeat
# here: reading a key back from a dict-of-lists field is a separate open
# wrong result.
class Store:
    def __init__(self):
        self.candidates = {}
        self.count = 0
        self.check = 0

    def fingerprint(self, mode: int, n: int) -> tuple[int, int]:
        if mode == 1:
            return 0, 0
        high = (n * 2654435761 + 12345) % 4294967296
        low = (n * 40503 + 777) % 4294967296
        first = high * (1 << 32) + low
        second = low * (1 << 32) + high
        return first, second

    def find_existing(self, domain: int, hi: int, lo: int) -> int:
        self.check = (
            self.check * 31 + hi % 1000003 + lo % 1000033
        ) % 1000000007
        key = (domain, hi, lo, 0)
        for token in self.candidates.get(key, ()):
            return token
        return 0

    def intern(self, domain: int, mode: int, n: int, fingerprint: tuple[int, int] | None = None) -> int:
        if fingerprint is None:
            hi, lo = self.fingerprint(mode, n)
        else:
            hi, lo = fingerprint
        existing = self.find_existing(domain, hi, lo)
        if existing != 0:
            return existing
        self.count += 1
        token = 1000 + self.count
        self.candidates.setdefault((domain, hi, lo, n), []).append(token)
        return token


    def intern_one(self, domain: int, mode: int, n: int) -> int:
        return self.intern(domain, mode, n)


def root(mode: int, n: int):
    store = Store()
    a = store.intern_one(1, mode, n)
    b = store.intern_one(2, mode, n + 1)
    c = store.intern_one(3, mode, n + 2)
    return (a * 1000000 + b * 1000 + c) % 1000000007 + store.check

# INPUTS [{'mode': 1, 'n': 3}, {'mode': 0, 'n': 0}, {'mode': 0, 'n': 1}, {'mode': 0, 'n': 3}]
