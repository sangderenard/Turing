def root(a: int, b: int) -> int:
    if a != 0 and b != 0 and 100 // (a * b) > 2:
        return 1
    return 0
# INPUTS [{'a':0,'b':3},{'a':2,'b':0},{'a':2,'b':3},{'a':10,'b':10}]
