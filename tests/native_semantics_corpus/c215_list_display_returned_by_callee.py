def single(n: int) -> list[int]:
    return [n]


def grown(n: int) -> list[int]:
    out = [n, 2]
    out.append(n * 3)
    return out


def root(n: int) -> int:
    t = single(n)
    u = grown(n)
    return len(t) * 10 + t[0] + len(u) * 1000 + (u[0] + u[1] + u[2]) * 100000

# INPUTS [{'n': 3}, {'n': -1}, {'n': 0}]
