# A user class named ``T`` constructs that class; the bare name is not
# the transpose alias.
class T:
    def __init__(self, v: int):
        self.v = v


def root(n: int) -> int:
    t = T(n)
    return t.v + 1

# INPUTS [{'n': 3}, {'n': -2}]
