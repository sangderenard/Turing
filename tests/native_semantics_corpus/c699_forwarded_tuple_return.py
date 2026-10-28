# ``return pack(a, b)`` returns the callee's whole tuple: every member,
# in order, with its own type.
def pack(a: float, b: float):
    return (a + b, a * b, 3)

def root(a: float, b: float):
    return pack(a, b)
# INPUTS [{'a': 1.5, 'b': 2.0}, {'a': -3.0, 'b': 0.5}]
