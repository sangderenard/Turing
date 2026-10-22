def add_one(xs: list[int]) -> None:
    xs.append(1)


def root(n: int) -> int:
    xs = [n, 4]
    add_one(xs)
    return len(xs) * 100 + xs[2]

# INPUTS [{'n': 1}, {'n': 5}]
