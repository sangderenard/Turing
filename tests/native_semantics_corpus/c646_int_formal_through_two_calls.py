# An int formal passed through two calls keeps its dtype in the formal-shape
# ledger / call ABI; the result is a negative int.
def inner(x: int, y: int) -> int:
    return x * y - 500
def outer(a: int) -> int:
    return inner(a, a + 1) + inner(a - 3, 2)
def root(n: int) -> int:
    return outer(n) - outer(n + 1)
# INPUTS [{'n': 1}, {'n': -4}, {'n': 25}]
