def edges(n: int) -> list[int]:
    r = []
    if n < 0:
        return r
    r.append(n * 2)
    return r


def root(n: int) -> int:
    found = edges(n)
    total = 0
    for x in found:
        total = total + x
    return total + len(found)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 5}]
