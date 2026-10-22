def helper(x: int, y: int) -> int:
    if x > y:
        return x - y
    return y * 2


def build(m: int) -> list[int]:
    out = []
    for i in range(m):
        out.append(i * i)
    return out


def pair(x: int, y: int) -> tuple[int, int]:
    return x // y, x % y


def first_over(x: int, y: int) -> int:
    for i in range(10):
        if i * y > x:
            return i
    return -1


def total(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def push(xs: list[int], v: int) -> None:
    xs.append(v)


def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def head(xs: list[int]) -> int:
    return xs[0]


def width(xs: list[int]) -> int:
    return len(xs)


def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def pick(xs: list[int], i: int) -> int:
    return xs[i]


def root(n: int) -> int:
    xs0 = [3, 1] + [n, 6]
    xs1 = [4, 4, 1]
    b2 = n > 4 or n < -7
    return (len(xs0) * 10 + xs0[2]) + ((1 if n in xs1 else 0) + (2 if 4 in xs1 else 0)) + (int(b2) + (3 if not b2 else 0))
# INPUTS [{'n': 1}, {'n': 5}, {'n': -8}]
