def root(n: int) -> int:
    d0 = {1: n, 2: 3}
    del d0[1]
    d1 = {1: n, 3: 5}
    s1 = 0
    for k1, w1 in d1.items():
        s1 += k1 * 10 + w1
    return len(d0) * 1000 + s1

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
