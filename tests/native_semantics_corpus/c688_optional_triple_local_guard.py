def root(a: int, b: int) -> int:
    if a > b:
        t = (a - b, a + b, a * b)
    else:
        t = None
    if t is None:
        return -7
    lo, hi, prod = t
    return lo * 1000 + hi * 10 + prod
# INPUTS [{'a':5,'b':2},{'a':2,'b':5},{'a':3,'b':3},{'a':-1,'b':-4}]
