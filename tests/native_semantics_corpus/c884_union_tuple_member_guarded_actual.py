# The c883 shape with a second callsite: ``preflighted`` passes a caller
# supplied pair, guarded into [0, U64_MAX] by a raising check, as ``intern``'s
# optional ``fingerprint``.  ``hi, lo`` then meet the constant ``0, 0``
# (int64), the computed uint64 combine and the guarded pair at one formal;
# the arm takes the width holding the interval joined over every actual.
U64_MAX = 18446744073709551615


class Store:
    def __init__(self, mode: int):
        self.mode = mode
        self.last = 0

    def fingerprint(self, n: int) -> tuple[int, int]:
        if self.mode == 0:
            return 0, 0
        high = (n * 2654435761 + 12345) % 4294967296
        low = (n * 40503 + 777) % 4294967296
        return high * (1 << 32) + low, low * (1 << 32) + high

    def find(self, n: int, hi: int, lo: int) -> int:
        return (hi % 1000003 + lo % 1000033 + n) % 1000000007

    def intern(self, n: int, fingerprint: tuple[int, int] | None = None) -> int:
        if fingerprint is None:
            hi, lo = self.fingerprint(n)
        else:
            hi, lo = fingerprint
        self.last = (self.last * 31 + self.find(n, hi, lo)) % 1000000007
        return self.last

    def preflighted(self, n: int, given_hi: int, given_lo: int) -> int:
        fingerprint_hi = given_hi * 4294967296 + given_lo
        fingerprint_lo = given_lo * 4294967296 + given_hi
        if fingerprint_hi < 0 or fingerprint_hi > U64_MAX or fingerprint_lo < 0 or fingerprint_lo > U64_MAX:
            raise ValueError("invalid preflight fingerprint")
        return self.intern(n, (fingerprint_hi, fingerprint_lo))


def root(mode: int, n: int):
    store = Store(mode)
    a = store.intern(n)
    b = store.preflighted(n, 2000000000 + n, 1500000000 + n)
    return a * 1000 + b % 1000

# INPUTS [{'mode': 0, 'n': 3}, {'mode': 1, 'n': 1}, {'mode': 1, 'n': 1000}]
