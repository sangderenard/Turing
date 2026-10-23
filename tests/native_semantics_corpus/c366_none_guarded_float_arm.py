def root(x: float, flag: int) -> float:
    s = None if flag > 0 else x
    t = None if s is None else s * 0.5
    total = 1.0
    if t is not None:
        total += t
    return total
# INPUTS [{'x':3.0,'flag':1},{'x':3.0,'flag':0}]
