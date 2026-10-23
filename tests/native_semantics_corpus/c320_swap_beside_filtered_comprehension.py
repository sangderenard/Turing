def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def root(n: int) -> int:
    xs0 = [i0 * 4 for i0 in range(n % 5 + 1) if i0 % 2 == 0]
    xs1 = [n, 5]
    swap(xs1)
    c2 = sum(i2 * n for i2 in range(4))
    return (len(xs0) * 100 + sum(xs0)) + (xs1[0] * 10 + xs1[1]) + (c2)
# INPUTS [{'n':0},{'n':2},{'n':-3},{'n':7}]
