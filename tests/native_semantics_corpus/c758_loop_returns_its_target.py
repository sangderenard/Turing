# A loop whose body only returns the loop variable returns the first
# element; an empty iterable falls through to the code after the loop.
def first(xs: list[int]) -> int:
    for t in xs:
        return t
    return -1


def root(n: int) -> int:
    e = [n]
    e.pop()
    return first([n + 1, 2]) * 10 + first(e)
# INPUTS [{'n': 3}, {'n': -6}]
