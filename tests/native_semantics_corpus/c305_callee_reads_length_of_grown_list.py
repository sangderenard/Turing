def summary(xs: list[int]) -> int:
    return len(xs) * 1000 + xs[len(xs) - 1] * 10 + xs[0]


def root(n: int) -> int:
    xs = [n]
    for i in range(n % 4 + 1):
        xs.append(i * 3)
    return summary(xs)

# INPUTS [{'n': 1}, {'n': 6}, {'n': -2}]
