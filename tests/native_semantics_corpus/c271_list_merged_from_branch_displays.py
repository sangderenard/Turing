def merged(n: int) -> int:
    if n > 2:
        t = [n, 1]
    else:
        t = [5, 6]
    return len(t) * 10 + t[0]


def merged_then_append(n: int) -> int:
    if n > 2:
        t = [n, 1]
    elif n > 0:
        t = [5, 6]
    else:
        t = [8, n]
    t.append(7)
    return len(t) * 10 + t[0] + t[2] * 100 + t[1] * 1000


def merged_with_alias(n: int) -> int:
    a = [1, 2]
    if n > 2:
        t = a
    else:
        t = [3, 4]
    return len(a) * 10 + len(t) + t[0] * 100


def root(n: int) -> int:
    u = [n, 2] if n > 2 else [5, 6]
    s = 0
    for i in range(3):
        if i == n % 3:
            t = [i, n]
        else:
            t = [7, 8]
        s += len(t) * 10 + t[0]
    return merged(n) + merged_then_append(n) * 100 + merged_with_alias(n) * 10000000 + s * 10000000000 + (len(u) * 10 + u[0]) * 10000000000000

# INPUTS [{'n': 3}, {'n': 1}, {'n': -2}]
