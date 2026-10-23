def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def width(xs: list[int]) -> int:
    return len(xs)


def root(n: int) -> int:
    xs0 = [5, -3, 8]
    ys1 = [6]
    grow(ys1, 2)
    return int(any([x0 > n for x0 in xs0])) * 10 + width(ys1) * 100 + ys1[-1]
# INPUTS [{'n':-3},{'n':0},{'n':9}]
