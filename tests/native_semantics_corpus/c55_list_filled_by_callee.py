def fill(dst: list[int], n: int) -> None:
    i = 0
    while i < n:
        dst.append(i * 2)
        i += 1


def root(n: int) -> int:
    xs = []
    fill(xs, n)
    total = 0
    for x in xs:
        total += x
    return len(xs) * 1000 + total
# INPUTS [{'n':0},{'n':1},{'n':6},{'n':50}]
