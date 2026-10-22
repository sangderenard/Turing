def fixed(n: int) -> list[int]:
    xs = [n, 1]
    return xs


def grown(n: int) -> list[int]:
    xs = []
    xs.append(n)
    xs.append(1)
    return xs


def root(n: int) -> int:
    t = fixed(n)
    t.append(9)
    u = grown(n)
    u.append(7)
    u.pop(0)
    return (len(t) * 10 + t[2] + t[0] * 100) + (len(u) * 10 + u[0] + u[1] * 100) * 10000

# INPUTS [{'n': 3}, {'n': -1}, {'n': 0}]
