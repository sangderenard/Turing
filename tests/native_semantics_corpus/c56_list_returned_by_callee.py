def make(n: int) -> list[int]:
    out = []
    i = 0
    while i < n:
        out.append(i + 1)
        i += 1
    return out


def root(n: int) -> int:
    ys = make(n)
    total = 0
    for y in ys:
        total += y
    return len(ys) * 1000 + total
# INPUTS [{'n':0},{'n':1},{'n':6},{'n':50}]
