def root(k: int) -> float:
    rep = [0.5, 0.25]
    if k <= 2:
        pass
    else:
        rep = [k * 0.25, 1.5, 2.0]
    return rep[-1] + len(rep) + rep[0]
# INPUTS [{'k':0},{'k':3},{'k':5}]
