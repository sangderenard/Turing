def compact(n: int):
    out = []
    for i in range(n):
        out.append(i + 10)
    if len(out) > 2:
        return out, True
    return [n, n + 1], False


def root(n: int) -> int:
    t, big = compact(n)
    return len(t) * 1000 + t[0] * 10 + (1 if big else 0)

# INPUTS [{"n": 0}, {"n": 1}, {"n": 3}, {"n": 6}]
