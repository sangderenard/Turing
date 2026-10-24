def root(k: int) -> int:
    rep = [1]
    if k > 2:
        rep = [k, 1, 2]
    return sum(rep) * 10 + len(rep) + max(rep)
# INPUTS [{'k':0},{'k':3},{'k':5}]
