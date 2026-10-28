# A forwarded tuple forwarded again keeps its members and their order.
def pair(a: int, b: int):
    return (b - a, a * b)

def relay(a: int, b: int):
    return pair(a + 1, b)

def root(a: int, b: int):
    return relay(a, b)
# INPUTS [{'a': 1, 'b': 2}, {'a': -4, 'b': 7}]
