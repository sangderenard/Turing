def root(n: int, t: int, w: int) -> int:
    # A nested arm and a while body rebind parameters before any read.
    if n > 0:
        if n > 2:
            t = n + 100
    k = 0
    while k < n:
        w = k + 10
        k += 1
    return t * 1000 + w

# INPUTS [{'n': 0, 't': 7, 'w': 4}, {'n': 1, 't': 7, 'w': 4}, {'n': 3, 't': 7, 'w': 4}]
