# An int record field projected inside a loop stays int64 (no float64 default
# on the projection cast), including a negative accumulation.
class Acc:
    def __init__(self, base: int):
        self.base = base
        self.step = 3
def root(n: int) -> int:
    a = Acc(n)
    t = 0
    for i in range(4):
        t += a.base * a.step - i
    return t - 1000
# INPUTS [{'n': 2}, {'n': -5}, {'n': 40}]
