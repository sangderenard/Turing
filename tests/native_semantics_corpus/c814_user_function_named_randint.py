# A user ``randint`` is the user's function, not the random source.
def randint(a: int, b: int) -> int:
    return a * 100 + b


def root(n: int) -> int:
    return randint(n, 4)

# INPUTS [{'n': 3}, {'n': -2}]
