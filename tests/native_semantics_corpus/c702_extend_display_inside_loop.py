# extend by a display of loop values, every iteration.
def root(n: int) -> int:
    xs = [0]
    i = 0
    while i < n:
        xs.extend([i, i + 1])
        i += 1
    return len(xs) * 100 + xs[len(xs) - 1]
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
