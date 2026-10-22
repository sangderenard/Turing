def root(n: int) -> int:
    xs = [1, 7, 8]
    return (1 if n in xs else 0) + (2 if 7 not in xs else 0) + len(xs)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 8}]
