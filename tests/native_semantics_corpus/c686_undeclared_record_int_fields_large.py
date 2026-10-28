# An int field of a record with no authored field declaration keeps int64
# through the receiver column (no float64 arena default): 2**55 + 1 is not
# representable in float64.
class Acc:
    def __init__(self, base: int):
        self.base = base
        self.step = 3
    def total(self, k: int) -> int:
        return self.base * self.step + k
def root(n: int, k: int) -> int:
    a = Acc(n)
    return a.total(k) - 1
# INPUTS [{'n': 36028797018963969, 'k': 5}, {'n': -7, 'k': 2}]
