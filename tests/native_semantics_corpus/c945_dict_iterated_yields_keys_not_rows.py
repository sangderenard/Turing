# Iterating a dict yields its keys.  The loop target of an unannotated
# dict took the class of its rows (here float), so ``k * 3`` was computed
# in float64 and the int program returned a float.
def root(n: int) -> int:
    d = {}
    for i in range(n):
        d[i * 3] = i * 0.5
    total = 0
    for k in d:
        total += k * 3
    return total
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
