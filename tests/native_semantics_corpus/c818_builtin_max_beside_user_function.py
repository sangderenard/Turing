# Builtin ``max`` stays the builtin when the program defines other functions.
def bump(x: int) -> int:
    return x + 1


def root(n: int) -> int:
    return max(bump(n), 5)

# INPUTS [{'n': 3}, {'n': 9}]
