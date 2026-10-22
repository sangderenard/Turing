def root(n: int) -> int:
    xs = [9, 8]
    for i in range(n):
        xs.reverse()
        xs.append(i)
    return xs[0] * 100 + xs[1] * 10 + len(xs)

# INPUTS [{'n': 1}, {'n': 4}]
