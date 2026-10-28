def pick(n: int) -> list[int]:
    if n > 2:
        x = [n, 1, 2]
    else:
        x = [4, 5, 6]
    return x
def root(n: int) -> int:
    t = pick(n)
    t.append(5)
    u = pick(n + 1)
    return len(t) * 1000 + t[0] * 10 + t[3] + u[0] * 100000
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
