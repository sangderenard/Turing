def root(k: int) -> int:
    d = {1: 10, 2: 20, 3: 30}
    if k in d:
        return d[k]
    return -1
# INPUTS [{'k':2},{'k':5}]
