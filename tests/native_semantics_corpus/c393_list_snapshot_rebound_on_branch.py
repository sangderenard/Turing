def root(k: int) -> int:
    rep = [7, 7]
    if k > 2:
        xs = [k]
        for i in range(k):
            xs.append(i * 3)
        rep = list(xs)
    return len(rep) * 100 + rep[0] * 10 + rep[-1]
# INPUTS [{'k':0},{'k':3},{'k':5}]
