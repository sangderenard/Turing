# The object is constructed in either arm, then used after the join.
class K:
    def __init__(self, scale: int):
        self.scale = scale
    def mul(self, x: int) -> int:
        return self.scale * x
def root(c: int, x: int) -> int:
    if c > 0:
        k = K(c)
    else:
        k = K(-c + 1)
    return k.mul(x)
# INPUTS [{'c': 3, 'x': 5}, {'c': -7, 'x': 2}, {'c': 0, 'x': 4}]
