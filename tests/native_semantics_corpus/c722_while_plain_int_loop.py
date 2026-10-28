def root(n: int, seed: int) -> int:
    acc = seed
    i = 0
    while i < n:
        acc = acc * 2 + i
        i += 1
    return acc
# INPUTS [{'n': 0, 'seed': 3}, {'n': 5, 'seed': -2}]
