def root(x: float, n: int) -> float:
    acc = 0.0
    k = 0
    while k < n:
        acc = acc + x / (k + 1)
        k += 1
    if acc > 2.0:
        acc = acc - 1.0
    return acc * 2.0
# INPUTS [{'x':1.5,'n':4},{'x':-2.0,'n':3},{'x':0.25,'n':0}]
