def maybe(x: float, flag: int):
    return x * 0.5 if flag > 0 else None

def wrap(x: float, flag: int):
    return maybe(x, flag)

def root(x: float, flag: int) -> float:
    a = wrap(x, flag)
    b = wrap(x, flag - 1)
    t = 0.0
    if a is not None:
        t += a
    if b is None:
        t += 100.0
    return t
# INPUTS [{'x':3.0,'flag':1},{'x':3.0,'flag':0},{'x':3.0,'flag':2}]
