def root(n: int) -> int:
    # A deleted key is not visited: iteration skips rows whose live flag a
    # delete cleared, and a re-stored key is visited once, last.
    d = {1: 10, 2: 20, 3: 30}
    del d[2]
    total = 0
    for k in d:
        total = total * 10 + k
    d[2] = n
    for k in d:
        total = total * 10 + k + d[k]
    return total

# INPUTS [{'n': 5}, {'n': 0}]
