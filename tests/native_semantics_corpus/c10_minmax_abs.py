def root(a: float, b: float) -> float:
    return max(a, b) - min(a, b) + abs(a - 2.0 * b)
# INPUTS [{'a':1.0,'b':3.5},{'a':-4.0,'b':-1.0}]
