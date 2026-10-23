def root(n: int) -> int:
    d = {}
    for i in range(4):
        d[i % 2] = d.get(i % 2, 0) + n
    e = {}
    for i in range(6):
        e[i % 3] = e.get(i % 3, 1) * 2 + n
    return d[0] * 10 + d[1] + (e[0] + e[1] * 10 + e[2] * 100) * 1000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
