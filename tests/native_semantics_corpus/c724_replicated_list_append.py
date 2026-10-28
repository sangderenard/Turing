def root(n: int) -> int:
    a = [0] * 3
    a.append(n)
    return len(a) * 100 + a[3]
# INPUTS [{'n': 3}, {'n': 8}]
