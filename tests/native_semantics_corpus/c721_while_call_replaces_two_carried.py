def step(acc: int, k: int) -> int:
    return acc * 2 + k


def stretch(acc: float, k: int) -> float:
    return acc + k * 0.5


def root(n: int, seed: int):
    acc = seed
    fine = 0.25
    i = 0
    while i < n:
        acc = step(acc, i)
        fine = stretch(fine, i)
        i += 1
    return acc * 1000.0 + fine
# INPUTS [{'n': 0, 'seed': 3}, {'n': 1, 'seed': 3}, {'n': 5, 'seed': -2}]
