def root(n: int) -> int:
    d = {}
    for i in range(n):
        d[i % 3] = d.get(i % 3, 0) + i
    total = 0
    for k in range(4):
        if k in d:
            total = total * 100 + d[k]
    return total

# INPUTS [{'n': 2}, {'n': 7}]
