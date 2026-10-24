# Keyword construction of the crossed-field class, read through a method.
class Q:
    def __init__(self, a: int, b: int):
        self.b = a
        self.a = b
    def get(self) -> int:
        return self.a * 100 + self.b
def root(x: int, y: int) -> int:
    return Q(b=x, a=y).get() + Q(x, y).a
# INPUTS [{'x': 3, 'y': 5}, {'x': -7, 'y': 2}]
