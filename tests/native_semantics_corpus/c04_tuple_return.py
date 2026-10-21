def divmod2(a: int, b: int):
    q = a // b
    r = a - q * b
    return q, r

def root(a: int, b: int) -> int:
    q, r = divmod2(a, b)
    return q * 100 + r
# INPUTS [{'a':17,'b':5},{'a':-7,'b':3},{'a':9,'b':9}]
