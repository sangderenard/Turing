# A loop over a comprehension of ints: the loop's result port is int, not the
# descriptor's float default.
def root(n: int, k: int) -> int:
    t = 0
    for y in [x * n + k for x in [2, 5, 9]]:
        t += y
    u = 1
    for z in [w * k for w in [3, 4]]:
        u = u * 10 + z
    return t * 100 + u
# INPUTS [{'n': 3, 'k': 2}, {'n': -4, 'k': 7}, {'n': 0, 'k': -1}]
