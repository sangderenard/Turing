def step(acc: int, k: int) -> int:
    return acc * 2 + k


def root(n: int, seed: int) -> int:
    acc = seed
    i = 0
    while i < n:
        acc = step(acc, i)
        i += 1
    return acc
# INPUTS [{'n': 0, 'seed': 3}, {'n': 1, 'seed': 3}, {'n': 5, 'seed': -2}]
