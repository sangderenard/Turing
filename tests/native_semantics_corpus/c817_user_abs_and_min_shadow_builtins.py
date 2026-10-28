# Module-level ``abs`` and ``min`` shadow the builtins; each call runs the
# program's own body.
def abs(x: int) -> int:
    return x * 2


def min(a: int, b: int) -> int:
    return a + b


def root(n: int) -> int:
    return abs(n) + min(n, 7)

# INPUTS [{'n': 3}, {'n': -4}]
