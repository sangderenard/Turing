def same(a: list[int]) -> list[int]:
    return a


def root(n: int) -> int:
    a = [1, 2, 3]
    t = same(a)
    t[1] = n
    a[2] = n + 10
    return a[0] + a[1] * 10 + a[2] * 100 + t[1] * 1000 + t[2]
# INPUTS [{'n': 0}, {'n': 1}, {'n': 7}]
