def root(t: int) -> float:
    xs = [2.5, t * 1.5]
    s = 0
    for v in xs:
        s += v
    return s
# INPUTS [{'t':3}, {'t':-2}]
