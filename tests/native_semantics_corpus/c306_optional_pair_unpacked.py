def maybe_pair(x: int, y: int) -> tuple[int, int] | None:
    if x > y:
        return None
    return (x + 1, y * 2)


def root(n: int, m: int) -> int:
    f = maybe_pair(n, m) if m > 0 else None
    if f is None:
        return -1
    a, b = f
    return a * 100 + b

# INPUTS [{'n': 1, 'm': 3}, {'n': 5, 'm': 3}, {'n': 1, 'm': 0}, {'n': -2, 'm': 4}]
