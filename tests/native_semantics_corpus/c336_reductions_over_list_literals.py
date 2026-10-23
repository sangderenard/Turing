def root(n: int) -> int:
    xs = [1] * 3
    return sum([1, 1, 1]) * 1000 + sum(xs) * 100 + max([2, 7, 5]) * 10 + min([3, 1, 4]) + n

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
