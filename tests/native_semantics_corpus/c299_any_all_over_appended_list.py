def root(n: int) -> int:
    xs = [n, 2]
    xs.append(n - 3)
    return int(any(xs)) * 10 + int(all(xs))

# INPUTS [{'n': 0}, {'n': 3}, {'n': 1}, {'n': -5}]
