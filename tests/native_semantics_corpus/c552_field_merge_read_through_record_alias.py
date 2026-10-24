# A field merge on a record with a second name (j = k): the method call
# through j reads the merged field, and the argument a keeps its value.
class K:
    def __init__(self, scale: int):
        self.scale = scale
    def mul(self, x: int) -> int:
        return self.scale * x
def root(a: int, x: int) -> int:
    k = K(a)
    j = k
    if x > 2:
        k.scale = k.scale + 100
    return j.mul(x) + k.scale + a
# INPUTS [{'a': 3, 'x': 5}, {'a': -7, 'x': 2}, {'a': 1, 'x': 3}]
