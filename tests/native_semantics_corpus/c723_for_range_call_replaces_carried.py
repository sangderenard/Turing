def step(acc: int, i: int) -> int:
    return acc * 2 + i


def root(n: int, seed: int) -> int:
    acc = seed
    for i in range(n):
        acc = step(acc, i)
    return acc
# INPUTS [{'n': 5, 'seed': -2}, {'n': 0, 'seed': 3}, {'n': 3, 'seed': 1}]
