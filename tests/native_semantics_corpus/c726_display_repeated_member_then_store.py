# One value named twice in a display is two members, each its own row.
def root(n: int) -> int:
    a = [n, 7, n, 7]
    a[1] = n + 1
    return len(a) * 100 + a[0] + a[3] + a[2]
# INPUTS [{'n': 3}, {'n': 4}]
