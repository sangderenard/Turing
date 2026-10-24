# __init__ stores its parameters into the other-named fields.
class Q:
    def __init__(self, a: int, b: int):
        self.b = a
        self.a = b
def root(x: int, y: int) -> int:
    q = Q(x, y)
    return q.a * 100 + q.b
# INPUTS [{'x': 3, 'y': 5}, {'x': -7, 'y': 2}]
