def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def repeated(n: int) -> int:
    xs0 = [1] * (n % 4 + 1)
    b1 = n > 3 or n < 3
    xs2 = [n, 3]
    swap(xs2)
    return (len(xs0) * 10 + xs0[0]) + int(b1) * 100 + (xs2[0] * 1000 + xs2[1])


def looped(n: int) -> int:
    xs0 = []
    k0 = n % 5 + 2
    while k0 > 0:
        xs0.append(k0)
        k0 -= 2
    c1 = 1 if 0 < n < 10 else 2
    xs2 = [n, 6]
    swap(xs2)
    return (len(xs0) * 10 + xs0[0]) + c1 * 100 + (xs2[0] * 1000 + xs2[1])


def root(n: int) -> int:
    return repeated(n) + looped(n) * 1000000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}]
