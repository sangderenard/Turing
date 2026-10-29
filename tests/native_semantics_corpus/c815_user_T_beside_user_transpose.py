# User functions ``T`` and ``transpose`` side by side each call their own
# body.
def transpose(x: int) -> int:
    return x - 1


def T(x: int) -> int:
    return x * 10


def root(n: int) -> int:
    return T(n) + transpose(n)

# INPUTS [{'n': 3}, {'n': -2}]
