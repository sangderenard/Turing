def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def head(xs: list[int]) -> int:
    return xs[0]


def width(xs: list[int]) -> int:
    return len(xs)


def root(n: int) -> int:
    xs = [n, 4]
    ys = [0]
    grow(ys, 2)
    return head(xs) + width(ys) * 10 + ys[-1]

# INPUTS [{'n': 1}, {'n': 5}]
