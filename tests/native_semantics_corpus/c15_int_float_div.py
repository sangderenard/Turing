def root(a: int, b: int) -> float:
    q = a // b
    r = a % b
    t = a / b
    return q + r * 0.001 + t
# INPUTS [{'a':7,'b':2},{'a':-7,'b':2},{'a':9,'b':-4}]
