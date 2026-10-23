def first_negative(xs: list[int]) -> int:
    for x in xs:
        if x < 0:
            return x
    return len(xs)


def short_length(xs: list[int]) -> int:
    if len(xs) > 3:
        return 0
    return len(xs)


def root(n: int) -> int:
    ys = [n, 1]
    if n > 0:
        ys.append(n)
    return first_negative([5, n, -1, n - 2]) * 100 + short_length(ys) * 10 + short_length([1, 2, 3, n])

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
