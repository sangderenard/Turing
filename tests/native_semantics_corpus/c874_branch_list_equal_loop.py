def root(n: int) -> int:
    s = 0
    for i in range(4):
        if i == n % 4:
            t = [i, n, 3]
        else:
            t = [7, 8, i]
        t.append(i)
        for v in t:
            s += v
        s += len(t) * 10 + t[0]
    return s
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
