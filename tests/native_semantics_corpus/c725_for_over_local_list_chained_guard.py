def root(n: int) -> int:
    xs = [1, 5, 9]
    r = 0
    for x in xs:
        if x <= n < x + 4 <= 12:
            r += x
    return r
# INPUTS [{'n': 0}, {'n': 5}, {'n': 9}]
