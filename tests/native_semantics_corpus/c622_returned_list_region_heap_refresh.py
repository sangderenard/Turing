def make(n: int):
    values = []
    for i in range(n):
        values.append(i + 3)
    return values, len(values)


def root(n: int) -> int:
    values, count = make(n)
    return values[0] * 100 + count if count else 0


# INPUTS [{"n": 0}, {"n": 1}, {"n": 3}]
