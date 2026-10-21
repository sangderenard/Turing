def root(a: int, b: int) -> int:
    c = a if a > b else b
    d = 1 if c % 2 == 0 else -1
    return c * d
# INPUTS [{'a':4,'b':3},{'a':2,'b':5},{'a':6,'b':6}]
