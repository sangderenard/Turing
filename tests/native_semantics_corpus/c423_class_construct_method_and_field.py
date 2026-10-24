# An ordinary class built in the program; a method and a field read of it.
class K:
    def __init__(self, scale: int):
        self.scale = scale
    def mul(self, x: int) -> int:
        return self.scale * x
def root(a: int, x: int) -> int:
    k = K(a)
    return k.mul(x) + k.scale
# INPUTS [{'a': 3, 'x': 5}, {'a': -7, 'x': 2}, {'a': 1152921504606846977, 'x': 1}]
