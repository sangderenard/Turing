def same(a: list[int]) -> list[int]:
    return a


def root(n: int) -> int:
    a = [1]
    t = same(a)
    for i in range(n):
        a.append(i * 2)
    t.append(7)
    return len(a) * 100 + len(t) * 10 + sum(a) + sum(t)
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
