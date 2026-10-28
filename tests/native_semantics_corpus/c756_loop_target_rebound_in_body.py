# A loop target the body rebinds reads the rebound value for the rest of
# that iteration, not the raw iteration element.
def root(n: int) -> int:
    total = 0
    for alias in [n, n + 1, n]:
        alias = alias * 2
        total += alias
    return total
# INPUTS [{'n': 3}, {'n': -4}]
