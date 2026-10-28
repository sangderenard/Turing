# A [7] * 5 list read at a runtime index is resident storage, not a literal.
def root(n: int) -> int:
    a = [7] * 5
    return a[n % 5] + len(a)
# INPUTS [{'n': 3}, {'n': 8}]
