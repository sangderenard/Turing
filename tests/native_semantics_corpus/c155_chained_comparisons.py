def root(n: int) -> int:
    r = 0
    if 0 < n < 10:
        r += 1
    if n == 5 or n > 8 and n % 2 == 0:
        r += 10
    if not (n >= 3):
        r += 100
    return r

# INPUTS [{'n': 5}, {'n': 10}, {'n': 2}, {'n': -1}]
