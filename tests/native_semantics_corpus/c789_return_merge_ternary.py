def pick(a: list[int], n: int) -> list[int]:
    return a if n > 2 else [7, n, 2]


def root(n: int) -> int:
    a = [n, 1]
    t = pick(a, n)
    u = pick(a, 5 - n)
    t.append(5)
    u.append(6)
    return len(a) * 1000 + len(t) * 100 + len(u) * 10 + t[0] + u[-1]
# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}, {'n': 3}, {'n': 4}, {'n': 9}]
