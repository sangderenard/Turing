def root(n: int) -> int:
    r = 0
    if n in [6, 3, 4]:
        r = 1
    if n in (2, 5):
        r += 10
    return r

# INPUTS [{'n': 3}, {'n': 5}, {'n': 7}]
