def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def root(n: int) -> int:
    c0 = 1 if n % 2 == 0 and n > 1 else 7
    xs1 = [n, 3]
    swap(xs1)
    xs2 = [i2 * 1 for i2 in range(n % 5 + 1) if i2 % 2 == 0]
    return (c0) + (xs1[0] * 10 + xs1[1]) + (len(xs2) * 100 + sum(xs2))
# INPUTS [{'n':0},{'n':2},{'n':-3},{'n':7}]
