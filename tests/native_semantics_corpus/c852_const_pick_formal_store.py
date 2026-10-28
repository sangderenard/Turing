def pick(a: list[int], k: int) -> list[int]:
    return a if k < 3 else [9, 9, 9]


def root(n: int) -> int:
    a = [1, 2, 3]
    t = pick(a, 1)
    t[0] = n
    return len(a)*100+len(t)*10+t[0]+a[0]
# INPUTS [{'n': 1}, {'n': 5}]
