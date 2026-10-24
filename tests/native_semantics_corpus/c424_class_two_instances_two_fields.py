# Two instances of a two-field ordinary class, arguments swapped.
class P:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def dot(self, a: int, b: int) -> int:
        return self.x * a + self.y * b
def root(x: int, y: int) -> int:
    p = P(y, x)
    q = P(x - 1, 2)
    return p.dot(3, 5) - q.dot(7, y) + p.x - q.y
# INPUTS [{'x': 3, 'y': 5}, {'x': -7, 'y': 2}, {'x': 0, 'y': 0}]
