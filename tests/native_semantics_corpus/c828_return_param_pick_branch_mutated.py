def pick(a: list[int], k: int) -> list[int]:
    return a if k > 3 else [9, 9, 9]


def root(k: int) -> int:
    a = [1, 2, 3]
    t = pick(a, k)
    u = pick(a, 5)
    t.append(5)
    u[0] = k
    return len(a) * 1000 + len(t) * 100 + len(u) * 10 + a[0] + t[-1]
# INPUTS [{'k': 0}, {'k': 3}, {'k': 4}, {'k': 9}]
