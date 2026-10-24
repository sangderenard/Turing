# A field written in the caller after construction, read by a method.
class K:
    def __init__(self, scale: int):
        self.scale = scale
    def mul(self, x: int) -> int:
        return self.scale * x
def root(a: int, x: int) -> int:
    k = K(a)
    if x > 2:
        k.scale = k.scale + 100
    return k.mul(x) + k.scale
# INPUTS [{'a': 3, 'x': 5}, {'a': -7, 'x': 2}, {'a': 1, 'x': 3}]
