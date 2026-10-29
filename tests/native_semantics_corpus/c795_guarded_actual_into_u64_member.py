# Token Orchard's preflighted intern, reduced: ``preflighted`` builds a
# fingerprint pair from caller-supplied ints, rejects it unless both halves
# lie in [0, U64_MAX] (``if ... < 0 or ... > U64_MAX: raise``), and passes it
# as a tuple to ``intern``, whose ``hi, lo`` members also receive the
# computed uint64 fingerprint.  The members settle uint64 from every actual
# reaching them; the guarded int64 pair enters them losing nothing because
# the guard proves it non-negative at that call.
U64_MAX = 18446744073709551615


class Store:
    def __init__(self):
        self.check = 0

    def fingerprint(self, mode: int, n: int) -> tuple[int, int]:
        if mode == 1:
            return 0, 0
        high = (n * 2654435761 + 12345) % 4294967296
        low = (n * 40503 + 777) % 4294967296
        return high * (1 << 32) + low, low * (1 << 32) + high

    def use(self, hi: int, lo: int) -> int:
        self.check = (self.check * 31 + hi % 1000003 + lo % 1000033) % 1000000007
        return self.check

    def intern(self, mode: int, n: int, fingerprint: tuple[int, int] | None = None) -> int:
        if fingerprint is None:
            hi, lo = self.fingerprint(mode, n)
        else:
            hi, lo = fingerprint
        return self.use(hi, lo)

    def preflighted(self, mode: int, n: int, given_hi: int, given_lo: int) -> int:
        fingerprint_hi = given_hi * 4294967296 + given_lo
        fingerprint_lo = given_lo * 4294967296 + given_hi
        if fingerprint_hi < 0 or fingerprint_hi > U64_MAX or fingerprint_lo < 0 or fingerprint_lo > U64_MAX:
            raise ValueError("invalid preflight fingerprint")
        return self.intern(mode, n, (fingerprint_hi, fingerprint_lo))


def root(mode: int, n: int):
    store = Store()
    a = store.intern(mode, n)
    b = store.preflighted(mode, n, 2000000000 + n, 1500000000 + n)
    c = store.intern(mode, n + 1)
    return a + b + c

# INPUTS [{'mode': 0, 'n': 3}, {'mode': 1, 'n': 1}, {'mode': 0, 'n': 1000}]
