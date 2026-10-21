def root(x: float, n: int) -> float:
    y = -x
    z = x ** 2 - y ** 3
    k = -n
    return z + k * 0.5 + (n ** 2)
# INPUTS [{'x':1.5,'n':3},{'x':-2.0,'n':-2}]
