def root(n: int) -> int:
    xs = [1, 2, 3, 4]
    s = {x % n for x in xs}
    return len(s)

# INPUTS [{'n': 1}, {'n': 2}, {'n': 3}, {'n': 5}]
