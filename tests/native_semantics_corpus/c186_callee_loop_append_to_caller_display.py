def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def root(n: int) -> int:
    xs = [n, 4]
    grow(xs, n % 3 + 1)
    return len(xs) * 100 + xs[-1] + xs[0]

# INPUTS [{'n': 1}, {'n': 5}]
