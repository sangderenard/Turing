def root(n: int) -> int:
    e = {}
    for i in range(5):
        e[i] = i * n
    del e[1]
    d = {1: 10, 2: 20, 3: n}
    del d[2]
    del d[3]
    return len(d) * 100 + len(e) * 10 + n

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
