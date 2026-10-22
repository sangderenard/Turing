def root(n: int) -> int:
    xs = [1, 5, 9]
    r = 0
    for x in xs:
        if x <= n < x + 4 <= 12:
            r += x
    k = 2 if 1 < n <= 6 else 3
    return r * 10 + k

# INPUTS [{'n': 0}, {'n': 5}, {'n': 9}]
