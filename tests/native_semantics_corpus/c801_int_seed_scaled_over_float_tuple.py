def root(t: int) -> float:
    s = 1
    for v in (t * 0.5, 2.0, 3.5):
        s = s * v
    return s
# INPUTS [{'t':3}, {'t':0}]
