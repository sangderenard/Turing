# A dict read with .get after a branch that stores into it: the join of the
# two versions is still the dict.
def root(n: int, m: int) -> int:
    d = {1: 5, 2: 6}
    if n > m:
        d[n] = m * 4
    return d.get(n, 100) * 3 + d.get(7, m) + len(d)
# INPUTS [{'n': 3, 'm': 1}, {'n': 1, 'm': 9}, {'n': 7, 'm': -2}]
