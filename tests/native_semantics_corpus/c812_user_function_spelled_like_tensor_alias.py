# A user function spelled like a catalog alias (``sum_``) is the user's
# function: it used to be renamed to ``sum`` and resolve to builtins.sum.
def sum_(a: int, b: int) -> int:
    return a * b


def root(n: int) -> int:
    return sum_(n, 3)

# INPUTS [{'n': 3}, {'n': -2}]
