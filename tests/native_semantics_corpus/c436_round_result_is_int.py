def root(n: int) -> int:
    # ``round`` of a float is an int, ties to even; the int sum reads it.
    s = 0
    for i in range(4):
        s = s + round(n * 0.5 + i * 0.5)
    x = n * 0.75
    return (round(x) + 3) * 100 + s

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}, {'n': 9}]
