def evens(n: int) -> list[int]:
    if n < 0:
        return []
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i * 3)
    return out


def root(n: int) -> int:
    t = evens(n)
    total = 0
    for v in t:
        total += v
    return len(t) * 1000 + total

# INPUTS [{"n": -2}, {"n": 0}, {"n": 5}, {"n": 9}]
