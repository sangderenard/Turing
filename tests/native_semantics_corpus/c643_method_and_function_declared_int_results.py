# Calls whose declared return class is int feed arithmetic beside a field
# written after construction.
class Box:
    def __init__(self, w: int):
        self.w = w
    def area(self, h: int) -> int:
        return self.w * h
def twice(v: int) -> int:
    return v + v
def root(a: int, b: int) -> int:
    bx = Box(a)
    if b > 2:
        bx.w = bx.w + 50
    return twice(bx.area(b)) - bx.w + twice(a)
# INPUTS [{'a': 4, 'b': 5}, {'a': -3, 'b': 1}, {'a': 0, 'b': 3}]
