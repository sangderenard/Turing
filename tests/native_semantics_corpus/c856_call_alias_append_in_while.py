def same(a: list[int]) -> list[int]:
    return a


def root(n: int) -> int:
    a = [n, 2, 3]
    t = same(a)
    i = 0
    while i < n:
        t.append(i)
        i += 1
    j = 0
    while j < n:
        a.append(j)
        t.append(j * 3)
        j += 1
    return len(a) * 100 + len(t) * 10 + sum(a) + sum(t)
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
