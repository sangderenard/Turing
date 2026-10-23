def root(n: int) -> int:
    d = {}
    for i in range(n):
        d[i % 3] = d.get(i % 3, 0) + 36028797018963969
    total = 0
    for k in range(4):
        if k in d:
            total = total + d[k] * (k + 1)
    return total

# INPUTS [{'n': 1}, {'n': 4}, {'n': 7}]
