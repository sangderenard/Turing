def root(n: int) -> int:
    # ``d[k]`` raises KeyError for an absent key, so a lookup must run only
    # where it is authored: behind its membership test, in its conditional
    # expression arm, and after the store or delete that decides its key.
    d = {1: n, 2: 5}
    t = 0
    for i in range(4):
        if i in d:
            t += d[i]
    u = d[n] if n in d else -1
    v = d[1] if n > 3 else d[2]
    d[7] = 3
    w = d[7]
    del d[2]
    x = d[n] if n == 1 else d.get(2, 8)
    return t * 1000 + u * 100 + v * 10 + w + x * 10000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}]
