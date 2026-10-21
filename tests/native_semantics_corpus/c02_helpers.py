def sq(x: float) -> float:
    return x * x

def hyp(a: float, b: float) -> float:
    s = sq(a) + sq(b)
    g = s * 0.5 + 0.5
    for _ in range(20):
        g = 0.5 * (g + s / g)
    return g

def root(a: float, b: float) -> float:
    return hyp(a, b) + sq(a - b)
# INPUTS [{'a':3.0,'b':4.0},{'a':1.5,'b':-2.0}]
