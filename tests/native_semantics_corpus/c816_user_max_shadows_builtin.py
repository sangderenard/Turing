# A module-level ``def max`` shadows the builtin, as in Python: the call runs
# the program's function.
def max(a: int, b: int) -> int:
    return a - b


def root(n: int) -> int:
    return max(n, 10)

# INPUTS [{'n': 3}, {'n': 20}]
