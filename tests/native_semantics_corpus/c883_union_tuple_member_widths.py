# Token Orchard's ``intern``: ``hi, lo = self.fingerprint(...)`` where one
# return is the constant ``0, 0`` (int64 by class) and the other the combine
# proven [0, 2**64-1] (uint64).  The pair's members meet two actuals of the
# SAME class ``builtins.int`` at one formal; the formal's arm takes the width
# holding the interval joined over every callsite reaching it (uint64), not
# the first callsite's, and no second callsite's arm is refused.
class Store:
    def __init__(self, mode: int):
        self.mode = mode
        self.last = 0

    def fingerprint(self, n: int, begin: int, end: int) -> tuple[int, int]:
        if self.mode == 0:
            return 0, 0
        high = (n * 2654435761 + 12345 + begin) % 4294967296
        low = (n * 40503 + 777 + end) % 4294967296
        return high * (1 << 32) + low, low * (1 << 32) + high

    def find(self, n: int, hi: int, lo: int) -> int:
        return (hi % 1000003 + lo % 1000033 + n) % 1000000007

    def intern(self, n: int, fingerprint: tuple[int, int] | None = None) -> int:
        if fingerprint is None:
            hi, lo = self.fingerprint(n, 0, n)
        else:
            hi, lo = fingerprint
        self.last = (self.last * 31 + self.find(n, hi, lo)) % 1000000007
        return self.last


    def intern_one(self, n: int) -> int:
        return self.intern(n)


def root(mode: int, n: int):
    store = Store(mode)
    a = store.intern_one(n)
    return a

# INPUTS [{'mode': 0, 'n': 3}, {'mode': 1, 'n': 1}, {'mode': 1, 'n': 1000}]
