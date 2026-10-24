# A method that reads (never stores) its receiver keeps the caller's field read.
class C:
    def __init__(self, n: int):
        self.n = n
    def pick(self) -> int:
        if self.n > 3:
            return self.n * 2
        return self.n - 1
def root(n: int) -> int:
    return C(n).pick()
# INPUTS [{'n': 3}, {'n': 4}, {'n': -2}]
