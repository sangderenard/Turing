def root(n: int) -> int:
    d = {1: 10, 2: 20}
    d.update({1: 5})
    r = d[1] + n
    e = {1: 10, 2: 20}
    e.update({2: n, 3: 7})
    r += e[2] * 100 + len(e) * 1000 + e[3]
    f = {1: 10, 2: 20}
    if n > 0:
        f.update({2: n})
    r += f[2] * 10000
    t = 0
    for i in range(n):
        t += d[1]
    return r + t * 1000000

# INPUTS [{'n': 3}, {'n': -1}, {'n': 0}]
