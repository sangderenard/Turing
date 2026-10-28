def root(n: int) -> int:
    a = [0] * 4
    for i in range(4):
        a[i] = i * n
    return a[3] + a[1]
# INPUTS [{'n': 3}, {'n': 8}]
