def pair(n: int) -> int:
    if n > 2:
        t = [n, 1]
    else:
        t = [5, 6]
    t.append(7)
    t.append(n)
    return len(t) * 100 + t[0] * 10 + t[3]


def plain(n: int) -> int:
    if n > 0:
        t = [n, 4, 2]
    else:
        t = [8, 9, 1]
    return len(t) * 10 + t[2]


def root(n: int) -> int:
    return pair(n) + plain(n) * 10000

# INPUTS [{'n': 3}, {'n': 1}, {'n': -2}, {'n': 6}]
