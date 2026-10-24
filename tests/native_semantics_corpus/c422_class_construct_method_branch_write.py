# An ordinary class constructed inside a program, then a method that writes
# its field in a branch (refused on main: no complete constructor schema).
class S:
    def __init__(self, n: int):
        self.n = n
    def bump(self, c: int) -> int:
        self.n = 0
        if c > 0:
            self.n = 1
        return self.n
def root(c: int) -> int:
    return S(5).bump(c)
# INPUTS [{'c': 3}, {'c': -1}, {'c': 0}]
