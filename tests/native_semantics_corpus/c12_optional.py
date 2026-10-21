def maybe(x: float):
    if x < 0.0:
        return None
    return x * 2.0

def root(x: float) -> float:
    v = maybe(x)
    if v is None:
        return -1.0
    return v + 1.0
# INPUTS [{'x':2.0},{'x':-3.0}]
