def root(n: int) -> int:
    if n > 0:
        s = {1, n}
    else:
        s = {2, 3}
    s.add(9)
    u = {1, 2} if n > 2 else {n, 3}
    t = 0
    for i in range(3):
        if i == n % 3:
            w = {i, n}
        else:
            w = {7, 8}
        t += len(w) * 10 + int(7 in w)
    return (len(s) * 10 + int(1 in s) + int(9 in s) * 100) + (len(u) * 10 + int(3 in u)) * 1000 + t * 100000

# INPUTS [{'n': 3}, {'n': 1}, {'n': -1}, {'n': 0}]
