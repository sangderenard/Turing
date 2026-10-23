def root(n: int) -> int:
    r = 0
    for x in (1, 5, 9):
        if x <= n:
            r += x
    f = 0
    for b in (True, False, True):
        if b:
            f += 1
    return r * 10 + f

# INPUTS [{'n': 0}, {'n': 5}, {'n': 9}]
