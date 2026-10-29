def same(a: list[int]) -> list[int]:
    return a


def root(n: int) -> int:
    a = [n, 2, 3]
    t = same(a)
    for i in range(n):
        t.append(i)
    a.append(9)
    return len(a) * 100 + len(t) * 10 + a[-1] + t[-1] + a[0] + sum(t)
# INPUTS [{'n': 0}, {'n': 1}, {'n': 5}]
