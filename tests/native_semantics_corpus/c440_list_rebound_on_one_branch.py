def root(k: int) -> int:
    rep = [9]
    if k > 2:
        rep = [k, 1, 2]
    return len(rep) * 10 + rep[0]
# INPUTS [{'k':0},{'k':3},{'k':5}]
